"""Conservative model-visible groundability checks for DPO candidates.

The frozen evaluator establishes that a prediction differs from canonical gold.
This module answers a separate question: whether every preference-bearing gold
value is supported by the user query plus the tool schema shown to the model.
External entity-to-ID knowledge and undocumented API encodings never pass.
"""

from __future__ import annotations

import calendar
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

try:
    from src.eval.analyze_errors import _schema_at_path, _tool_index, _user_query
except ModuleNotFoundError:
    from analyze_errors import (  # type: ignore[no-redef]
        _schema_at_path,
        _tool_index,
        _user_query,
    )


GROUNDABILITY_VERSION = "model-visible-groundability-v1"

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_ORDINAL_PAGES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}
_COUNTRY_CODES = {
    "australia": "AU",
    "brazil": "BR",
    "canada": "CA",
    "china": "CN",
    "france": "FR",
    "germany": "DE",
    "great britain": "GB",
    "india": "IN",
    "italy": "IT",
    "japan": "JP",
    "netherlands": "NL",
    "norway": "NO",
    "spain": "ES",
    "uk": "GB",
    "united kingdom": "GB",
    "us": "US",
    "usa": "US",
    "united states": "US",
}
_GENERIC_MAPPING_TOKENS = {
    "about",
    "argument",
    "available",
    "category",
    "code",
    "data",
    "default",
    "field",
    "filter",
    "find",
    "from",
    "get",
    "include",
    "information",
    "input",
    "language",
    "limit",
    "list",
    "market",
    "name",
    "number",
    "option",
    "page",
    "parameter",
    "region",
    "request",
    "result",
    "search",
    "string",
    "tool",
    "type",
    "value",
    "with",
}
_OPAQUE_PATH_PARTS = {
    "channelid",
    "code",
    "id",
    "newspaperid",
    "symbol",
    "ticker",
    "uri",
    "uuid",
}
_AMBIGUOUS_LITERAL_STRINGS = {"a", "all", "an", "and", "or", "s"}


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _number_forms(value: int | float) -> set[str]:
    forms = {str(value)}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    return forms


def _literal_in_query(value: Any, query: str) -> bool:
    normalized_query = _normalized_text(query)
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return re.search(
            rf"(?<!\w){str(value).casefold()}(?!\w)", normalized_query
        ) is not None
    if isinstance(value, (int, float)):
        return any(
            re.search(
                rf"(?<![\d.]){re.escape(form)}(?!\d)(?!\.\d)",
                normalized_query,
            )
            is not None
            for form in _number_forms(value)
        )
    if isinstance(value, str):
        literal = _normalized_text(value)
        if literal in _AMBIGUOUS_LITERAL_STRINGS:
            return any(
                quoted in query.casefold()
                for quoted in (f"'{literal}'", f'"{literal}"', f"`{literal}`")
            )
        if re.fullmatch(r"[\w-]+", literal, flags=re.UNICODE):
            return re.search(
                rf"(?<!\w){re.escape(literal)}(?!\w)", normalized_query
            ) is not None
        return literal in normalized_query
    if isinstance(value, list):
        return bool(value) and all(_literal_in_query(child, query) for child in value)
    if isinstance(value, dict):
        return bool(value) and all(
            _literal_in_query(child, query) for child in value.values()
        )
    return False


def _schema_text(schema: dict[str, Any], *, include_default: bool = False) -> str:
    def without_defaults(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_defaults(child)
                for key, child in value.items()
                if include_default or key != "default"
            }
        if isinstance(value, list):
            return [without_defaults(child) for child in value]
        return value

    return _normalized_text(json.dumps(without_defaults(schema), ensure_ascii=False))


def _schemas_for_diff(
    case: dict[str, Any], diff: dict[str, Any]
) -> list[dict[str, Any]]:
    sample_id = str(case.get("source_sample_id", "unknown"))
    index = _tool_index(case.get("relevant_tools"), sample_id=sample_id)
    return [
        schema
        for tool in index.get(diff["tool"], [])
        if (schema := _schema_at_path(tool, diff["path"])) is not None
    ]


def _is_optional_argument(case: dict[str, Any], diff: dict[str, Any]) -> bool:
    sample_id = str(case.get("source_sample_id", "unknown"))
    index = _tool_index(case.get("relevant_tools"), sample_id=sample_id)
    root = diff["path"].split(".", 1)[0]
    options = index.get(diff["tool"], [])
    if not options:
        return False
    for tool in options:
        function = tool.get("function", {})
        parameters = function.get("parameters", {})
        required = parameters.get("required", [])
        if isinstance(required, list) and root in required:
            return False
    return True


def _looks_opaque(path: str, value: Any) -> bool:
    compact_path = re.sub(r"[^a-z0-9]", "", path.casefold())
    if any(part in compact_path for part in _OPAQUE_PATH_PARTS):
        return True
    if isinstance(value, str):
        if re.match(r"^[a-z][a-z0-9+.-]*:[^\s]+$", value, flags=re.IGNORECASE):
            return True
        if len(value) >= 10 and re.search(r"[A-Za-z]", value) and re.search(
            r"\d", value
        ):
            return True
    return False


def _has_explicit_opaque_risk(path: str, value: Any) -> bool:
    """Avoid substring collisions such as ``uri`` inside ``query``."""
    compact_path = re.sub(r"[^a-z0-9]", "", path.casefold())
    path_risk = (
        compact_path in _OPAQUE_PATH_PARTS
        or compact_path.endswith(("id", "uuid", "uri", "ticker", "symbol", "code"))
    )
    if path_risk:
        return True
    if isinstance(value, str):
        if re.match(r"^[a-z][a-z0-9+.-]*:[^\s]+$", value, flags=re.IGNORECASE):
            return True
        if (
            re.fullmatch(r"[A-Za-z0-9_.:+/-]{10,}", value)
            and re.search(r"[A-Za-z]", value)
            and re.search(r"\d", value)
        ):
            return True
    return False


def _percentages(query: str) -> list[float]:
    return [
        float(match.group(1))
        for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:%|percent\b)", query, re.I)
    ]


def _percentage_evidence(
    value: Any, query: str, schemas: list[dict[str, Any]]
) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    schema_text = " ".join(_schema_text(schema) for schema in schemas)
    for percent in _percentages(query):
        if re.search(r"hundredths?|basis points?", schema_text) and float(value) == percent * 100:
            return "schema_documented_percentage_hundredths"
        if re.search(r"decimal|fraction|between 0 and 1", schema_text) and float(value) == percent / 100:
            return "schema_documented_percentage_fraction"
    return None


def _dozen_evidence(value: Any, query: str) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number_pattern = "|".join([r"\d+"] + list(_WORD_NUMBERS))
    for match in re.finditer(rf"(?<!\w)({number_pattern})\s+dozen\b", query, re.I):
        token = match.group(1).casefold()
        count = int(token) if token.isdigit() else _WORD_NUMBERS[token]
        if float(value) == count * 12:
            return "deterministic_dozen_conversion"
    return None


def _page_number(query: str) -> int | None:
    normalized = _normalized_text(query)
    for word, number in _ORDINAL_PAGES.items():
        if re.search(rf"\b{word}\s+page\b", normalized):
            return number
    match = re.search(r"\bpage\s+(\d+)\b", normalized)
    return int(match.group(1)) if match else None


def _pagination_evidence(
    value: Any,
    query: str,
    schemas: list[dict[str, Any]],
    diff: dict[str, Any],
    case: dict[str, Any],
) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if diff["path"].split(".")[-1].casefold() not in {"offset", "skip"}:
        return None
    schema_text = " ".join(_schema_text(schema) for schema in schemas)
    if not re.search(r"skip|offset", schema_text):
        return None
    page = _page_number(query)
    if page is None:
        return None
    gold_calls = case.get("canonical_gold", {}).get("tool_calls", [])
    call_index = diff.get("gold_call_index")
    if not isinstance(call_index, int) or not 0 <= call_index < len(gold_calls):
        return None
    arguments = gold_calls[call_index].get("function", {}).get("arguments", {})
    limit = arguments.get("limit")
    if (
        isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and _literal_in_query(limit, query)
        and float(value) == (page - 1) * float(limit)
    ):
        return "deterministic_page_limit_to_offset"
    return None


def _iso_country_evidence(
    value: Any, query: str, schemas: list[dict[str, Any]]
) -> str | None:
    if not isinstance(value, str):
        return None
    schema_text = " ".join(_schema_text(schema) for schema in schemas)
    if not re.search(r"iso\s*-?\s*3166", schema_text):
        return None
    normalized_query = _normalized_text(query)
    for phrase, code in _COUNTRY_CODES.items():
        if code.casefold() != value.casefold():
            continue
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_query):
            return "schema_declared_iso_3166_country_mapping"
    return None


def _date_evidence(value: Any, query: str, schemas: list[dict[str, Any]]) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    schema_text = " ".join(_schema_text(schema) for schema in schemas)
    if not re.search(r"yyyy\s*[-/]\s*mm\s*[-/]\s*dd|iso\s*-?\s*8601", schema_text):
        return None
    year, month, day = (int(part) for part in value.split("-"))
    names = {calendar.month_name[month], calendar.month_abbr[month]}
    normalized_query = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", query, flags=re.I)
    return (
        "schema_documented_date_format"
        if any(
            re.search(
                rf"\b{re.escape(name)}\s+{day}(?:,)?\s+{year}\b",
                normalized_query,
                re.I,
            )
            for name in names
        )
        else None
    )


def _schema_mapping_evidence(
    value: Any, query: str, schemas: list[dict[str, Any]]
) -> str | None:
    if value is None or value == "" or isinstance(value, (list, dict, bool)):
        return None

    def mapping_text(schema: dict[str, Any]) -> str:
        fragments: list[str] = []

        def collect(node: Any, *, key: str | None = None) -> None:
            if isinstance(node, dict):
                if "const" in node and any(
                    isinstance(node.get(label), str)
                    for label in ("title", "description")
                ):
                    label = " ".join(
                        str(node[name])
                        for name in ("title", "description")
                        if isinstance(node.get(name), str)
                    )
                    fragments.append(f"{node['const']} means {label}")
                for child_key, child in node.items():
                    if child_key == "default":
                        continue
                    collect(child, key=child_key)
            elif isinstance(node, list):
                for child in node:
                    collect(child, key=key)
            elif isinstance(node, str) and key in {"description", "title"}:
                fragments.extend(
                    sentence
                    for sentence in re.split(r"(?<=[.!?])\s+", node)
                    if not re.search(r"\bdefaults?\b", sentence, re.I)
                )

        collect(schema)
        return _normalized_text(" ".join(fragments))

    def lexical_variants(text: str) -> set[str]:
        normalized = _normalized_text(text)
        variants = {normalized}
        if normalized.endswith("s") and len(normalized) > 4:
            variants.add(normalized[:-1])
        return variants

    normalized_query = _normalized_text(query)
    for schema in schemas:
        text = mapping_text(schema)
        if not _literal_in_query(value, text):
            continue
        if isinstance(value, str) and any(
            re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized_query)
            for variant in lexical_variants(value)
        ):
            return "schema_explicit_option_or_mapping"
        value_text = _normalized_text(str(value))
        position = text.find(value_text)
        if position < 0:
            continue
        window = text[max(0, position - 80) : position + len(value_text) + 80]
        if not re.search(r"(?:means|indicates|represents|\bfor\b|[:=])", window):
            continue
        query_words = [
            token
            for token in re.findall(r"\b[^\W\d_]\w{3,}\b", normalized_query)
            if token not in _GENERIC_MAPPING_TOKENS
        ]
        phrases = {
            " ".join(query_words[index : index + width])
            for width in (2, 3, 4)
            for index in range(len(query_words) - width + 1)
        }
        if any(phrase in window for phrase in phrases):
            return "schema_explicit_option_or_mapping"
    return None


def _boolean_evidence(
    value: Any, query: str, schemas: list[dict[str, Any]]
) -> str | None:
    if not isinstance(value, bool):
        return None
    normalized_query = _normalized_text(query)
    schema_text = " ".join(_schema_text(schema) for schema in schemas)
    query_tokens = set(re.findall(r"\b[^\W\d_]\w{3,}\b", normalized_query))
    schema_tokens = set(re.findall(r"\b[^\W\d_]\w{3,}\b", schema_text))
    meaningful_overlap = {
        token
        for token in query_tokens & schema_tokens
        if token not in _GENERIC_MAPPING_TOKENS
    }
    if not meaningful_overlap:
        return None
    if value is False and re.search(
        r"\b(?:without|exclude|excluding|omit|omitting|disable|disabled|no)\b|do not|don't",
        normalized_query,
    ):
        return "query_explicit_negative_constraint"
    if value is True and re.search(
        r"\b(?:include|including|enable|enabled|with)\b", normalized_query
    ):
        return "query_explicit_positive_constraint"
    return None


def _leaf_evidence(
    value: Any,
    query: str,
    schemas: list[dict[str, Any]],
    diff: dict[str, Any],
    case: dict[str, Any],
) -> str | None:
    for resolver in (
        lambda: _percentage_evidence(value, query, schemas),
        lambda: _dozen_evidence(value, query),
        lambda: _pagination_evidence(value, query, schemas, diff, case),
        lambda: _iso_country_evidence(value, query, schemas),
        lambda: _date_evidence(value, query, schemas),
        lambda: _boolean_evidence(value, query, schemas),
    ):
        if evidence := resolver():
            return evidence
    if _literal_in_query(value, query):
        return "query_literal"
    if evidence := _schema_mapping_evidence(value, query, schemas):
        return evidence
    return None


def _flatten_leaves(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _flatten_leaves(child, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _flatten_leaves(child, f"{path}[{index}]")
        ]
    return [(path, value)]


def assess_diff_groundability(
    case: dict[str, Any], diff: dict[str, Any]
) -> dict[str, Any]:
    query = case.get("user_query")
    if not isinstance(query, str) or not query:
        raise TypeError(f"{case.get('source_sample_id')}: missing user_query")
    schemas = _schemas_for_diff(case, diff)
    if diff["type"] == "extra_argument" and not diff["gold_present"]:
        passed = bool(schemas) and _is_optional_argument(case, diff)
        return {
            "tool": diff["tool"],
            "path": diff["path"],
            "diagnostic": diff["type"],
            "passed": passed,
            "evidence": (
                "schema_supported_omission_of_semantically_rejected_extra"
                if passed
                else None
            ),
            "reason": (
                "Canonical omission is schema-valid; the prior semantic review "
                "established that the rejected extra argument is harmful."
                if passed
                else "Canonical omission is not established as schema-valid."
            ),
            "ungrounded_leaves": [],
        }

    if not diff["gold_present"]:
        return {
            "tool": diff["tool"],
            "path": diff["path"],
            "diagnostic": diff["type"],
            "passed": False,
            "evidence": None,
            "reason": "The preference-bearing canonical value is absent.",
            "ungrounded_leaves": [{"path": "$", "value": None}],
        }

    leaves = _flatten_leaves(diff["gold"])
    evidence = []
    ungrounded = []
    for leaf_path, value in leaves:
        leaf_evidence = _leaf_evidence(value, query, schemas, diff, case)
        if leaf_evidence is None:
            ungrounded.append(
                {
                    "path": leaf_path,
                    "value": value,
                    "looks_opaque": _looks_opaque(diff["path"], value),
                }
            )
        else:
            evidence.append(leaf_evidence)
    passed = bool(leaves) and not ungrounded
    if passed:
        reason = "Every canonical difference leaf has model-visible evidence."
    elif any(leaf["looks_opaque"] for leaf in ungrounded):
        reason = (
            "At least one canonical value requires an undocumented or external "
            "entity-to-identifier mapping."
        )
    else:
        reason = (
            "At least one canonical value is neither present in the query nor "
            "derivable through a documented deterministic schema transformation."
        )
    return {
        "tool": diff["tool"],
        "path": diff["path"],
        "diagnostic": diff["type"],
        "passed": passed,
        "evidence": sorted(set(evidence)),
        "reason": reason,
        "ungrounded_leaves": ungrounded,
    }


def assess_case_groundability(case: dict[str, Any]) -> dict[str, Any]:
    diffs = case.get("argument_diffs")
    if not isinstance(diffs, list) or not diffs:
        raise TypeError(f"{case.get('source_sample_id')}: invalid argument_diffs")
    assessments = [assess_diff_groundability(case, diff) for diff in diffs]
    passed = all(assessment["passed"] for assessment in assessments)
    evidence_counts = Counter(
        evidence
        for assessment in assessments
        for evidence in (
            [assessment["evidence"]]
            if isinstance(assessment["evidence"], str)
            else assessment["evidence"] or []
        )
    )
    return {
        "version": GROUNDABILITY_VERSION,
        "passed": passed,
        "decision": "grounded" if passed else "excluded_ungrounded",
        "policy": (
            "Every preference-bearing canonical difference must be a query "
            "literal, a deterministic query+schema transformation, an explicit "
            "schema option/mapping, or a schema-valid omission previously judged "
            "to remove a harmful extra argument. Defaults alone never establish "
            "entity-to-ID/code/URI/ticker mappings."
        ),
        "diff_assessments": assessments,
        "failed_diff_count": sum(not assessment["passed"] for assessment in assessments),
        "evidence_counts": dict(sorted(evidence_counts.items())),
    }


def triage_gold_sample_groundability(sample: dict[str, Any]) -> dict[str, Any]:
    """Apply cheap three-way rule triage to a prompt/gold sample before rollout.

    Unlike :func:`assess_case_groundability`, this function has no model
    prediction and therefore cannot know which values will become preference
    differences.  ``AUTO_ACCEPT`` only means that every canonical argument leaf
    matched one of the narrow verifier-backed patterns implemented here;
    ``AUTO_REJECT`` is reserved for obvious opaque/external mappings or non-tool
    targets; everything else remains ``SEMANTIC_UNKNOWN``.  None of these labels
    replaces post-rollout difference-level groundability adjudication.
    """
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise TypeError("canonical sample has no valid sample_id")
    query = _user_query(sample.get("messages"), sample_id=sample_id)
    tools = sample.get("tools")
    tool_index = _tool_index(tools, sample_id=sample_id)
    assistant = sample.get("assistant")
    if not isinstance(assistant, dict):
        raise TypeError(f"{sample_id}: canonical assistant must be an object")
    calls = assistant.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise TypeError(f"{sample_id}: assistant.tool_calls must be an array")
    if not calls:
        return {
            "version": GROUNDABILITY_VERSION,
            "scope": "prompt_gold_prescreen",
            "decision": "AUTO_REJECT",
            "reason": "non_tool_call_target",
            "call_type": "text",
            "gold_call_count": 0,
            "gold_argument_count": 0,
            "gold_leaf_count": 0,
            "missing_tool_definitions": [],
            "argument_assessments": [],
            "failed_argument_count": 0,
            "ungrounded_leaf_count": 0,
            "evidence_counts": {},
        }

    canonical_gold = {"content": assistant.get("content"), "tool_calls": calls}
    case = {
        "source_sample_id": sample_id,
        "user_query": query,
        "relevant_tools": tools,
        "canonical_gold": canonical_gold,
    }
    missing_tools: list[str] = []
    assessments: list[dict[str, Any]] = []
    gold_leaf_count = 0
    for call_index, wrapper in enumerate(calls):
        function = wrapper.get("function") if isinstance(wrapper, dict) else None
        if not isinstance(function, dict):
            raise TypeError(
                f"{sample_id}: assistant.tool_calls[{call_index}].function "
                "must be an object"
            )
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{sample_id}: assistant.tool_calls[{call_index}] has invalid name"
            )
        if not isinstance(arguments, dict):
            raise TypeError(
                f"{sample_id}: assistant.tool_calls[{call_index}] has invalid arguments"
            )
        if name not in tool_index:
            missing_tools.append(name)
        for path, value in arguments.items():
            if not isinstance(path, str) or not path:
                raise TypeError(
                    f"{sample_id}: assistant.tool_calls[{call_index}] has invalid "
                    "argument name"
                )
            gold_leaf_count += len(_flatten_leaves(value))
            assessments.append(
                assess_diff_groundability(
                    case,
                    {
                        "tool": name,
                        "gold_call_index": call_index,
                        "pred_call_index": None,
                        "path": path,
                        "type": "gold_argument_prescreen",
                        "gold_present": True,
                        "pred_present": False,
                        "gold": value,
                        "pred": None,
                    },
                )
            )

    failed = [assessment for assessment in assessments if not assessment["passed"]]
    evidence_counts = Counter(
        evidence
        for assessment in assessments
        for evidence in (
            [assessment["evidence"]]
            if isinstance(assessment["evidence"], str)
            else assessment["evidence"] or []
        )
    )
    ungrounded_leaves = [
        leaf
        for assessment in failed
        for leaf in assessment["ungrounded_leaves"]
    ]
    all_leaves_rule_verified = not missing_tools and not failed
    if missing_tools:
        decision = "AUTO_REJECT"
        reason = "missing_tool_definition"
    elif any(leaf["looks_opaque"] for leaf in ungrounded_leaves):
        decision = "AUTO_REJECT"
        reason = "opaque_or_external_mapping"
    elif failed:
        decision = "SEMANTIC_UNKNOWN"
        reason = "requires_semantic_groundability_judgment"
    else:
        decision = "AUTO_ACCEPT"
        reason = "all_gold_argument_leaves_match_verifier_backed_rules"
    return {
        "version": GROUNDABILITY_VERSION,
        "scope": "prompt_gold_prescreen",
        "decision": decision,
        "reason": reason,
        "all_leaves_rule_verified": all_leaves_rule_verified,
        "call_type": "single" if len(calls) == 1 else "multi",
        "gold_call_count": len(calls),
        "gold_argument_count": len(assessments),
        "gold_leaf_count": gold_leaf_count,
        "missing_tool_definitions": sorted(set(missing_tools)),
        "argument_assessments": assessments,
        "failed_argument_count": len(failed),
        "ungrounded_leaf_count": len(ungrounded_leaves),
        "evidence_counts": dict(sorted(evidence_counts.items())),
    }


def triage_candidate_difference_groundability(
    case: dict[str, Any],
) -> dict[str, Any]:
    """Three-way verifier-backed triage for a post-rollout structural case.

    ``PASS`` is deliberately narrower than the binary grounding assessment: it
    also rejects known representation/default equivalence and abstains on
    schema-valid extra arguments, nested structural alternatives, or string
    expansions that may be semantically equivalent.  ``UNKNOWN`` is an
    abstention, not a negative data-quality label.
    """
    preference_filter = case.get("preference_filter")
    if isinstance(preference_filter, dict) and preference_filter.get(
        "decision"
    ) == "excluded":
        return {
            "version": GROUNDABILITY_VERSION,
            "scope": "post_rollout_difference_triage",
            "decision": "REJECT",
            "reason": preference_filter.get("basis")
            or "preference_filter_exclusion",
            "groundability": None,
        }

    groundability = assess_case_groundability(case)
    if not groundability["passed"]:
        opaque = any(
            leaf.get("looks_opaque")
            for assessment in groundability["diff_assessments"]
            for leaf in assessment["ungrounded_leaves"]
        )
        return {
            "version": GROUNDABILITY_VERSION,
            "scope": "post_rollout_difference_triage",
            "decision": "REJECT" if opaque else "UNKNOWN",
            "reason": (
                "opaque_or_external_mapping"
                if opaque
                else "difference_requires_semantic_groundability_judgment"
            ),
            "groundability": groundability,
        }

    diagnostics = {diff["type"] for diff in case["argument_diffs"]}
    if "extra_argument" in diagnostics:
        return {
            "version": GROUNDABILITY_VERSION,
            "scope": "post_rollout_difference_triage",
            "decision": "UNKNOWN",
            "reason": "schema_valid_extra_argument_harmfulness_not_rule_proven",
            "groundability": groundability,
        }
    if "nested_structure_mismatch" in diagnostics:
        return {
            "version": GROUNDABILITY_VERSION,
            "scope": "post_rollout_difference_triage",
            "decision": "UNKNOWN",
            "reason": "nested_schema_valid_alternative_requires_semantic_review",
            "groundability": groundability,
        }

    query = case["user_query"]
    for diff, assessment in zip(
        case["argument_diffs"],
        groundability["diff_assessments"],
        strict=True,
    ):
        evidence = assessment["evidence"]
        evidence_set = (
            {evidence}
            if isinstance(evidence, str)
            else set(evidence or [])
        )
        if "query_literal" not in evidence_set:
            continue
        prediction = diff.get("pred")
        if _literal_in_query(prediction, query):
            return {
                "version": GROUNDABILITY_VERSION,
                "scope": "post_rollout_difference_triage",
                "decision": "UNKNOWN",
                "reason": "gold_and_rejected_values_both_appear_in_query",
                "groundability": groundability,
            }
        gold = diff.get("gold")
        schemas = _schemas_for_diff(case, diff)
        schema_text = " ".join(_schema_text(schema) for schema in schemas)
        enum_proves_choice = any(
            isinstance(schema.get("enum"), list)
            and gold in schema["enum"]
            and prediction in schema["enum"]
            for schema in schemas
        )
        rejected_matches_default = any(
            "default" in schema
            and (
                schema["default"] == prediction
                or (
                    isinstance(schema["default"], str)
                    and isinstance(prediction, str)
                    and _normalized_text(schema["default"])
                    == _normalized_text(prediction)
                )
            )
            for schema in schemas
        )
        if diff["type"] == "wrong_value" and rejected_matches_default:
            return {
                "version": GROUNDABILITY_VERSION,
                "scope": "post_rollout_difference_triage",
                "decision": "UNKNOWN",
                "reason": "rejected_matches_schema_default",
                "groundability": groundability,
            }
        if not enum_proves_choice and (
            _has_explicit_opaque_risk(diff["path"], gold)
            or re.search(
                r"\b(?:identifier|code|slug|symbol|ticker|uri|uuid)\b",
                schema_text,
            )
        ):
            return {
                "version": GROUNDABILITY_VERSION,
                "scope": "post_rollout_difference_triage",
                "decision": "UNKNOWN",
                "reason": "literal_identifier_or_code_mapping_requires_semantic_review",
                "groundability": groundability,
            }
        if isinstance(gold, str) and isinstance(prediction, str):
            normalized_gold = _normalized_text(gold)
            normalized_prediction = _normalized_text(prediction)
            alphanumeric_gold = re.sub(
                r"[^a-z0-9]+", "", normalized_gold, flags=re.IGNORECASE
            )
            alphanumeric_prediction = re.sub(
                r"[^a-z0-9]+", "", normalized_prediction, flags=re.IGNORECASE
            )
            if (
                normalized_gold != normalized_prediction
                and (
                    normalized_gold in normalized_prediction
                    or normalized_prediction in normalized_gold
                    or (
                        alphanumeric_gold
                        and alphanumeric_gold == alphanumeric_prediction
                    )
                )
            ):
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "possible_semantically_equivalent_string_expansion",
                    "groundability": groundability,
                }
            prediction_literal = _normalized_text(prediction)
            simple_inflections = {
                prediction_literal + "s",
                prediction_literal + "es",
                (
                    prediction_literal[:-1] + "ies"
                    if prediction_literal.endswith("y")
                    else prediction_literal
                ),
            }
            if any(
                re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", query.casefold())
                for variant in simple_inflections
                if variant
            ):
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "rejected_simple_inflection_also_appears_in_query",
                    "groundability": groundability,
                }
            if not enum_proves_choice and min(len(gold), len(prediction)) <= 3:
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "possible_name_to_short_code_equivalence",
                    "groundability": groundability,
                }
            gold_numbers = [
                abs(float(value))
                for value in re.findall(r"[-+]?\d+(?:\.\d+)?", gold)
            ]
            prediction_numbers = [
                abs(float(value))
                for value in re.findall(r"[-+]?\d+(?:\.\d+)?", prediction)
            ]
            if (
                gold_numbers
                and len(gold_numbers) == len(prediction_numbers)
                and sorted(gold_numbers) == sorted(prediction_numbers)
            ):
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "possible_equivalent_numeric_string_representation",
                    "groundability": groundability,
                }
            if not enum_proves_choice and SequenceMatcher(
                None, alphanumeric_gold, alphanumeric_prediction
            ).ratio() >= 0.75:
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "possible_alias_or_near_equivalent_string",
                    "groundability": groundability,
                }
            quoted_gold = any(
                quoted in query.casefold()
                for quoted in (
                    f"'{gold.casefold()}'",
                    f'"{gold.casefold()}"',
                    f"`{gold.casefold()}`",
                )
            )
            if (
                diff["type"] == "wrong_value"
                and not enum_proves_choice
                and len(gold) < 20
                and not quoted_gold
            ):
                return {
                    "version": GROUNDABILITY_VERSION,
                    "scope": "post_rollout_difference_triage",
                    "decision": "UNKNOWN",
                    "reason": "free_form_string_preference_not_rule_proven",
                    "groundability": groundability,
                }

    return {
        "version": GROUNDABILITY_VERSION,
        "scope": "post_rollout_difference_triage",
        "decision": "PASS",
        "reason": "all_differences_are_verifier_backed_without_known_ambiguity",
        "groundability": groundability,
    }

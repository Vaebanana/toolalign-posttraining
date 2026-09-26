"""Frozen Step 4 metrics for strict tool-use evaluation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from src.data.validate import collect_tool_validators

try:
    from .parser import ParseResult, ToolCall
except ImportError:  # Support ``python src/eval/evaluate.py``.
    from parser import ParseResult, ToolCall


PROTOCOL_VERSION = "toolalign-step4-v1"
GoldMode = Literal["tool", "text"]
PredictedMode = Literal["tool", "text", "invalid"]


@dataclass(frozen=True)
class SampleScore:
    """All sufficient statistics and flags for one evaluated sample."""

    gold_mode: GoldMode
    predicted_mode: PredictedMode
    gold_call_count: int
    predicted_call_count: int
    gold_call_kind: Literal["single", "multi", "text"]
    response_mode_correct: bool
    format_valid: bool
    tool_name_exact: bool
    tool_name_tp: int
    predicted_name_count: int
    gold_name_count: int
    argument_exact_matches: int
    name_matched_calls: int
    argument_exact: bool | None
    schema_valid: bool
    full_call_exact: bool
    no_tool_correct: bool | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_mode": self.gold_mode,
            "predicted_mode": self.predicted_mode,
            "gold_call_count": self.gold_call_count,
            "predicted_call_count": self.predicted_call_count,
            "gold_call_kind": self.gold_call_kind,
            "response_mode_correct": self.response_mode_correct,
            "format_valid": self.format_valid,
            "tool_name_exact": self.tool_name_exact,
            "tool_name_tp": self.tool_name_tp,
            "predicted_name_count": self.predicted_name_count,
            "gold_name_count": self.gold_name_count,
            "argument_exact_matches": self.argument_exact_matches,
            "name_matched_calls": self.name_matched_calls,
            "argument_exact": self.argument_exact,
            "schema_valid": self.schema_valid,
            "full_call_exact": self.full_call_exact,
            "no_tool_correct": self.no_tool_correct,
            "errors": list(self.errors),
        }


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """Normalize object key order/whitespace only; preserve values and arrays."""
    return json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def gold_tool_calls(assistant: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Read already-validated canonical gold calls without string parsing."""
    raw_calls = assistant.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise TypeError("canonical assistant.tool_calls must be an array")
    calls: list[ToolCall] = []
    for index, wrapper in enumerate(raw_calls):
        if (
            not isinstance(wrapper, dict)
            or wrapper.get("type") != "function"
            or not isinstance(wrapper.get("function"), dict)
        ):
            raise ValueError(f"gold tool call {index} has invalid wrapper")
        function = wrapper["function"]
        if set(function) != {"name", "arguments"}:
            raise ValueError(f"gold tool call {index} has invalid fields")
        name = function["name"]
        arguments = function["arguments"]
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError(f"gold tool call {index} has invalid value types")
        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


def _name_counter(calls: tuple[ToolCall, ...]) -> Counter[str]:
    return Counter(call.name for call in calls)


def _full_counter(calls: tuple[ToolCall, ...]) -> Counter[tuple[str, str]]:
    return Counter(
        (call.name, canonical_arguments(call.arguments)) for call in calls
    )


def _argument_match_counts(
    gold_calls: tuple[ToolCall, ...],
    predicted_calls: tuple[ToolCall, ...],
) -> tuple[int, int]:
    gold_by_name: dict[str, Counter[str]] = defaultdict(Counter)
    predicted_by_name: dict[str, Counter[str]] = defaultdict(Counter)
    for call in gold_calls:
        gold_by_name[call.name][canonical_arguments(call.arguments)] += 1
    for call in predicted_calls:
        predicted_by_name[call.name][canonical_arguments(call.arguments)] += 1

    names = set(gold_by_name) | set(predicted_by_name)
    exact = sum(
        sum((gold_by_name[name] & predicted_by_name[name]).values())
        for name in names
    )
    matched_names = sum(
        min(sum(gold_by_name[name].values()), sum(predicted_by_name[name].values()))
        for name in names
    )
    return exact, matched_names


def _schema_valid(
    parse_result: ParseResult,
    candidate_tools: list[dict[str, Any]],
) -> bool:
    if parse_result.mode == "invalid" or not parse_result.calls:
        return False
    definitions, definition_errors = collect_tool_validators(candidate_tools)
    if definition_errors:
        return False
    for call in parse_result.calls:
        matches = definitions.get(call.name, [])
        if len(matches) != 1:
            return False
        validator = matches[0][1]
        if next(validator.iter_errors(call.arguments), None) is not None:
            return False
    return True


def score_sample(
    gold_assistant: dict[str, Any],
    candidate_tools: list[dict[str, Any]],
    parse_result: ParseResult,
) -> SampleScore:
    """Score one prediction under the frozen Step 4 evaluation protocol."""
    gold_calls = gold_tool_calls(gold_assistant)
    predicted_calls = parse_result.calls
    gold_mode: GoldMode = "tool" if gold_calls else "text"
    if parse_result.mode in {"tool", "mixed"}:
        predicted_mode: PredictedMode = "tool"
    elif parse_result.mode == "text":
        predicted_mode = "text"
    else:
        predicted_mode = "invalid"

    gold_names = _name_counter(gold_calls)
    predicted_names = _name_counter(predicted_calls)
    name_intersection = gold_names & predicted_names
    name_tp = sum(name_intersection.values())
    tool_name_exact = gold_names == predicted_names
    argument_matches, matched_names = _argument_match_counts(
        gold_calls,
        predicted_calls,
    )
    argument_exact = (
        argument_matches == matched_names if matched_names > 0 else None
    )
    schema_valid = _schema_valid(parse_result, candidate_tools)
    format_valid = parse_result.strict_format_valid
    full_call_exact = (
        gold_mode == "tool"
        and format_valid
        and _full_counter(gold_calls) == _full_counter(predicted_calls)
    )
    no_tool_correct = predicted_mode == "text" if gold_mode == "text" else None

    errors: list[str] = []
    if predicted_mode != gold_mode:
        errors.append("wrong_mode")
    if gold_mode == "tool" and not format_valid:
        errors.append("format_error")
    if any(name not in gold_names for name in predicted_names):
        errors.append("wrong_tool")
    if any(gold_names[name] > predicted_names[name] for name in gold_names):
        errors.append("missing_tool")
    if any(predicted_names[name] > gold_names[name] for name in predicted_names):
        errors.append("extra_tool")
    if matched_names > 0 and argument_matches < matched_names:
        errors.append("argument_mismatch")
    if gold_mode == "tool" and not schema_valid:
        errors.append("schema_invalid")

    kind: Literal["single", "multi", "text"]
    if not gold_calls:
        kind = "text"
    elif len(gold_calls) == 1:
        kind = "single"
    else:
        kind = "multi"
    return SampleScore(
        gold_mode=gold_mode,
        predicted_mode=predicted_mode,
        gold_call_count=len(gold_calls),
        predicted_call_count=len(predicted_calls),
        gold_call_kind=kind,
        response_mode_correct=predicted_mode == gold_mode,
        format_valid=format_valid,
        tool_name_exact=tool_name_exact,
        tool_name_tp=name_tp,
        predicted_name_count=len(predicted_calls),
        gold_name_count=len(gold_calls),
        argument_exact_matches=argument_matches,
        name_matched_calls=matched_names,
        argument_exact=argument_exact,
        schema_valid=schema_valid,
        full_call_exact=full_call_exact,
        no_tool_correct=no_tool_correct,
        errors=tuple(errors),
    )


def rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def aggregate_scores(scores: list[SampleScore]) -> dict[str, Any]:
    """Aggregate sufficient statistics without defining a composite score."""
    tool_scores = [score for score in scores if score.gold_mode == "tool"]
    text_scores = [score for score in scores if score.gold_mode == "text"]
    tool_name_tp = sum(score.tool_name_tp for score in tool_scores)
    predicted_names = sum(score.predicted_name_count for score in tool_scores)
    gold_names = sum(score.gold_name_count for score in tool_scores)
    if gold_names:
        precision = tool_name_tp / predicted_names if predicted_names else 0.0
        recall = tool_name_tp / gold_names
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    else:
        precision = None
        recall = None
        f1 = None

    return {
        "primary": {
            "full_call_success_rate": rate(
                sum(score.full_call_exact for score in tool_scores),
                len(tool_scores),
            )
        },
        "secondary": {
            "no_tool_accuracy": rate(
                sum(score.no_tool_correct is True for score in text_scores),
                len(text_scores),
            )
        },
        "diagnostic": {
            "response_mode_accuracy": rate(
                sum(score.response_mode_correct for score in scores),
                len(scores),
            ),
            "tool_format_valid_rate": rate(
                sum(score.format_valid for score in tool_scores),
                len(tool_scores),
            ),
            "tool_name_exact_match": rate(
                sum(score.tool_name_exact for score in tool_scores),
                len(tool_scores),
            ),
            "tool_name_f1": {
                "value": f1,
                "precision": precision,
                "recall": recall,
                "true_positives": tool_name_tp,
                "predicted_calls": predicted_names,
                "gold_calls": gold_names,
            },
            "argument_exact_match": rate(
                sum(score.argument_exact_matches for score in tool_scores),
                sum(score.name_matched_calls for score in tool_scores),
            ),
            "schema_valid_rate": rate(
                sum(score.schema_valid for score in tool_scores),
                len(tool_scores),
            ),
        },
    }

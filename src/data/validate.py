"""Canonical candidates 的最终结构与 arguments/Schema 校验。

JSON Schema 语义完全交给 ``jsonschema``；本模块只负责 canonical 外围结构、
工具名解析、错误归类和最终状态判定，不执行任何修复。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jsonschema import SchemaError
from jsonschema.validators import validator_for

KEEP = "keep"
REPAIRED_KEEP = "repaired_keep"
MANUAL_REVIEW = "manual_review"
DROP = "drop"

CANONICAL_FIELDS = {
    "sample_id",
    "source",
    "messages",
    "tools",
    "assistant",
    "metadata",
}
MESSAGE_FIELDS = {"role", "content"}
TOOL_FIELDS = {"type", "function"}
TOOL_FUNCTION_FIELDS = {"name", "description", "parameters"}
ASSISTANT_FIELDS = {"content", "tool_calls"}
TOOL_CALL_FIELDS = {"type", "function"}
TOOL_CALL_FUNCTION_FIELDS = {"name", "arguments"}
METADATA_FIELDS = {"category", "subcategory", "task"}


def _error(
    issue: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {"issue": issue, "message": message, **details}


def _append_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def _validate_top_level(
    sample: Any,
    drop_errors: list[dict[str, Any]],
) -> bool:
    if not isinstance(sample, dict):
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "canonical sample must be an object",
            )
        )
        return False

    if set(sample) != CANONICAL_FIELDS:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "canonical top-level fields do not match the contract",
                missing_fields=sorted(CANONICAL_FIELDS - set(sample)),
                unexpected_fields=sorted(set(sample) - CANONICAL_FIELDS),
            )
        )

    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "sample_id must be a non-empty string",
                field="sample_id",
            )
        )

    if sample.get("source") not in {"xlam", "hermes"}:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "source must be xlam or hermes",
                field="source",
            )
        )

    return True


def _validate_messages_and_metadata(
    sample: dict[str, Any],
    drop_errors: list[dict[str, Any]],
) -> None:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "messages must be a non-empty array",
                field="messages",
            )
        )
    else:
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or set(message) != MESSAGE_FIELDS
                or message.get("role") not in {"system", "user"}
                or not isinstance(message.get("content"), str)
            ):
                drop_errors.append(
                    _error(
                        "invalid_canonical_structure",
                        "message does not match canonical structure",
                        message_index=index,
                    )
                )

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != METADATA_FIELDS:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "metadata does not match canonical structure",
                field="metadata",
            )
        )


def _collect_tools(
    tools: Any,
    drop_errors: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], Any]]]:
    """验证工具 wrapper/Schema，返回 name -> (definition, validator)。"""
    definitions: dict[str, list[tuple[dict[str, Any], Any]]] = {}

    if not isinstance(tools, list):
        drop_errors.append(
            _error(
                "invalid_tools_structure",
                "tools must be an array",
            )
        )
        return definitions

    for tool_index, tool in enumerate(tools):
        if (
            not isinstance(tool, dict)
            or set(tool) != TOOL_FIELDS
            or tool.get("type") != "function"
            or not isinstance(tool.get("function"), dict)
        ):
            drop_errors.append(
                _error(
                    "invalid_tools_structure",
                    "tool wrapper does not match canonical structure",
                    tool_index=tool_index,
                )
            )
            continue

        function = tool["function"]
        if set(function) != TOOL_FUNCTION_FIELDS:
            drop_errors.append(
                _error(
                    "invalid_tools_structure",
                    "tool function fields do not match the contract",
                    tool_index=tool_index,
                )
            )
            continue

        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(description, str)
            or not isinstance(parameters, dict)
        ):
            drop_errors.append(
                _error(
                    "invalid_tools_structure",
                    "tool name/description/parameters has invalid type",
                    tool_index=tool_index,
                )
            )
            continue

        try:
            validator_class = validator_for(parameters)
            validator_class.check_schema(parameters)
            validator = validator_class(parameters)
        except SchemaError as error:
            drop_errors.append(
                _error(
                    "invalid_tool_schema",
                    error.message,
                    tool_index=tool_index,
                    tool_name=name,
                    schema_path=list(error.path),
                )
            )
            continue

        definitions.setdefault(name, []).append((function, validator))

    return definitions


def collect_tool_validators(
    tools: Any,
) -> tuple[
    dict[str, list[tuple[dict[str, Any], Any]]],
    list[dict[str, Any]],
]:
    """Build sample-local JSON Schema validators for downstream evaluation.

    This public wrapper keeps schema interpretation shared with canonical data
    validation.  Callers receive structural/schema errors instead of repairing
    invalid definitions.
    """
    errors: list[dict[str, Any]] = []
    return _collect_tools(tools, errors), errors


def _validate_assistant(
    assistant: Any,
    tools_by_name: dict[str, list[tuple[dict[str, Any], Any]]],
    drop_errors: list[dict[str, Any]],
    schema_errors: list[dict[str, Any]],
) -> None:
    if not isinstance(assistant, dict) or set(assistant) != ASSISTANT_FIELDS:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "assistant does not match canonical structure",
                field="assistant",
            )
        )
        return

    content = assistant.get("content")
    tool_calls = assistant.get("tool_calls")
    if content is not None and not isinstance(content, str):
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "assistant.content must be string or null",
                field="assistant.content",
            )
        )
    if not isinstance(tool_calls, list):
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "assistant.tool_calls must be an array",
                field="assistant.tool_calls",
            )
        )
        return
    if content is not None and tool_calls:
        drop_errors.append(
            _error(
                "invalid_canonical_structure",
                "assistant cannot contain text and tool calls together",
            )
        )
    if content is None and not tool_calls:
        drop_errors.append(
            _error(
                "empty_assistant_output",
                "assistant must contain text or at least one tool call",
            )
        )

    for call_index, call in enumerate(tool_calls):
        if (
            not isinstance(call, dict)
            or set(call) != TOOL_CALL_FIELDS
            or call.get("type") != "function"
            or not isinstance(call.get("function"), dict)
        ):
            drop_errors.append(
                _error(
                    "invalid_tool_call_structure",
                    "tool call wrapper does not match canonical structure",
                    tool_call_index=call_index,
                )
            )
            continue

        function = call["function"]
        if set(function) != TOOL_CALL_FUNCTION_FIELDS:
            drop_errors.append(
                _error(
                    "invalid_tool_call_structure",
                    "tool call function fields do not match the contract",
                    tool_call_index=call_index,
                )
            )
            continue

        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            drop_errors.append(
                _error(
                    "invalid_tool_call_structure",
                    "tool call name must be a non-empty string",
                    tool_call_index=call_index,
                )
            )
            continue
        if not isinstance(arguments, dict):
            drop_errors.append(
                _error(
                    "invalid_arguments_type",
                    "tool call arguments must be an object",
                    tool_call_index=call_index,
                    tool_name=name,
                )
            )
            continue

        matches = tools_by_name.get(name, [])
        if not matches:
            drop_errors.append(
                _error(
                    "unknown_tool",
                    "tool call name was not found in tools",
                    tool_call_index=call_index,
                    tool_name=name,
                )
            )
            continue
        if len(matches) != 1:
            drop_errors.append(
                _error(
                    "ambiguous_tool_definition",
                    "tool call name resolves to multiple tool definitions",
                    tool_call_index=call_index,
                    tool_name=name,
                )
            )
            continue

        validator = matches[0][1]
        validation_errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: (
                tuple(str(part) for part in error.path),
                tuple(str(part) for part in error.schema_path),
                error.message,
            ),
        )
        for validation_error in validation_errors:
            schema_errors.append(
                _error(
                    "arguments_schema_mismatch",
                    validation_error.message,
                    tool_call_index=call_index,
                    tool_name=name,
                    validator=validation_error.validator,
                    argument_path=list(validation_error.path),
                    schema_path=list(validation_error.schema_path),
                )
            )


def validate_sample(
    sample: Any,
    repaired_sample_ids: set[str],
) -> tuple[dict[str, Any], bool]:
    """验证一条 canonical sample，返回 validation record 与是否可训练。"""
    drop_errors: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    top_level_is_object = _validate_top_level(sample, drop_errors)

    if top_level_is_object:
        _validate_messages_and_metadata(sample, drop_errors)
        tools_by_name = _collect_tools(sample.get("tools"), drop_errors)
        _validate_assistant(
            sample.get("assistant"),
            tools_by_name,
            drop_errors,
            schema_errors,
        )

    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    raw_source = sample.get("source") if isinstance(sample, dict) else None
    source = raw_source if isinstance(raw_source, str) else "unknown"
    if drop_errors:
        status = DROP
        errors = drop_errors + schema_errors
    elif schema_errors:
        status = MANUAL_REVIEW
        errors = schema_errors
    elif sample_id in repaired_sample_ids:
        status = REPAIRED_KEEP
        errors = []
    else:
        status = KEEP
        errors = []

    issues: list[str] = []
    for error in errors:
        _append_issue(issues, error["issue"])

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "source": source,
        "status": status,
        "issues": issues,
    }
    if errors:
        record["validation_errors"] = errors

    return record, status in {KEEP, REPAIRED_KEEP}


def summarize_validation(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """汇总四种最终状态与各来源计数。"""
    statuses = {KEEP, REPAIRED_KEEP, MANUAL_REVIEW, DROP}
    status_counts = {status: 0 for status in statuses}
    source_status_counts: dict[str, dict[str, int]] = {}
    total = 0

    for record in records:
        total += 1
        status = record["status"]
        source = record["source"]
        status_counts[status] += 1
        per_source = source_status_counts.setdefault(
            source,
            {name: 0 for name in statuses},
        )
        per_source[status] += 1

    return {
        "input_candidates": total,
        "status_counts": {
            KEEP: status_counts[KEEP],
            REPAIRED_KEEP: status_counts[REPAIRED_KEEP],
            MANUAL_REVIEW: status_counts[MANUAL_REVIEW],
            DROP: status_counts[DROP],
        },
        "final_usable": status_counts[KEEP] + status_counts[REPAIRED_KEEP],
        "excluded": status_counts[MANUAL_REVIEW] + status_counts[DROP],
        "by_source": {
            source: {
                KEEP: counts[KEEP],
                REPAIRED_KEEP: counts[REPAIRED_KEEP],
                MANUAL_REVIEW: counts[MANUAL_REVIEW],
                DROP: counts[DROP],
            }
            for source, counts in sorted(source_status_counts.items())
        },
    }

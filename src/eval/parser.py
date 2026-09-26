"""Strictly parse Qwen tool-call predictions without repairing model output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

ParseMode = Literal["tool", "text", "mixed", "invalid"]
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)
TAG_MARKERS = ("<tool_call", "</tool_call")


@dataclass(frozen=True)
class ToolCall:
    """Unified evaluation representation of one function call."""

    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ParseResult:
    """Structured prediction plus protocol status."""

    mode: ParseMode
    calls: tuple[ToolCall, ...]
    strict_format_valid: bool
    raw_text: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "calls": [call.to_dict() for call in self.calls],
            "strict_format_valid": self.strict_format_valid,
            "raw_text": self.raw_text,
            "errors": list(self.errors),
        }


def _reject_non_json_constant(value: str) -> Any:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _parse_call_body(body: str, index: int) -> ToolCall:
    try:
        value = json.loads(body, parse_constant=_reject_non_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"call_{index}:invalid_json") from error
    if not isinstance(value, dict):
        raise TypeError(f"call_{index}:body_not_object")
    if set(value) != {"name", "arguments"}:
        raise ValueError(f"call_{index}:invalid_fields")
    name = value["name"]
    arguments = value["arguments"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"call_{index}:invalid_name")
    if not isinstance(arguments, dict):
        raise TypeError(f"call_{index}:arguments_not_object")
    return ToolCall(name=name, arguments=arguments)


def parse_prediction(raw_prediction: str) -> ParseResult:
    """Parse one raw prediction using strict JSON and exact XML-like tags.

    No Python-literal fallback, coercion, trimming of values or other repair is
    performed.  Whitespace surrounding complete tool-call blocks is allowed.
    """
    if not isinstance(raw_prediction, str):
        raise TypeError("raw_prediction must be a string")
    if not raw_prediction.strip():
        return ParseResult(
            mode="invalid",
            calls=(),
            strict_format_valid=False,
            raw_text=raw_prediction,
            errors=("empty_prediction",),
        )

    matches = list(TOOL_CALL_PATTERN.finditer(raw_prediction))
    if not matches:
        if any(marker in raw_prediction for marker in TAG_MARKERS):
            return ParseResult(
                mode="invalid",
                calls=(),
                strict_format_valid=False,
                raw_text=raw_prediction,
                errors=("malformed_tool_call_tags",),
            )
        return ParseResult(
            mode="text",
            calls=(),
            strict_format_valid=False,
            raw_text=raw_prediction,
        )

    outside_parts: list[str] = []
    cursor = 0
    calls: list[ToolCall] = []
    errors: list[str] = []
    for index, match in enumerate(matches):
        outside_parts.append(raw_prediction[cursor : match.start()])
        cursor = match.end()
        try:
            calls.append(_parse_call_body(match.group(1), index))
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    outside_parts.append(raw_prediction[cursor:])
    outside = "".join(outside_parts)

    if any(marker in outside for marker in TAG_MARKERS):
        errors.append("malformed_tool_call_tags")
    if errors:
        return ParseResult(
            mode="invalid",
            calls=tuple(calls),
            strict_format_valid=False,
            raw_text=raw_prediction,
            errors=tuple(errors),
        )
    if outside.strip():
        return ParseResult(
            mode="mixed",
            calls=tuple(calls),
            strict_format_valid=False,
            raw_text=raw_prediction,
            errors=("prose_outside_tool_calls",),
        )
    return ParseResult(
        mode="tool",
        calls=tuple(calls),
        strict_format_valid=True,
        raw_text=raw_prediction,
    )

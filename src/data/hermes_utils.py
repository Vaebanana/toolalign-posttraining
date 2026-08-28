"""Hermes 审计与确定性修复共用的底层辅助函数。

这里的函数只负责生成可验证的解析/恢复候选和定位 Schema 问题；是否真正
修改数据由 repair 模块决定。这样 audit 与 repair 使用同一套证据生成逻辑。
"""

from __future__ import annotations

import ast
import json
import re
from json import JSONDecodeError
from typing import Any, Iterable


BOUNDARY_LITERAL_ESCAPE = "boundary_literal_escape"
PYTHON_LITERAL_SYNTAX = "python_literal_syntax"
UNCLASSIFIED_PARSE_FAILURE = "unclassified_parse_failure"
BOUNDARY_JSON_DIAGNOSTIC = "boundary_literal_escape_removed_json"
PYTHON_LITERAL_DIAGNOSTIC = "python_literal"

JSON_SCHEMA_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}

SYSTEM_TOOLS_PATTERN = re.compile(
    r"<tools>(.*?)</tools>",
    flags=re.DOTALL,
)
FUNCTION_WRAPPER_START_PATTERN = re.compile(
    r'\{\s*"type"\s*:\s*"function"'
)


def remove_boundary_literal_escapes(text: str) -> str:
    """只删除字符串首尾各一个字面量 ``\\n``，不碰正文内容。"""
    candidate = text

    if candidate.startswith(r"\n"):
        candidate = candidate[2:]

    if candidate.endswith(r"\n"):
        candidate = candidate[:-2]

    return candidate


def diagnose_tool_call_text(raw_content: str) -> dict[str, Any]:
    """按既定优先级生成 tool_call 的唯一诊断解析候选。"""
    has_boundary_literal_escape = (
        raw_content.startswith(r"\n")
        or raw_content.endswith(r"\n")
    )
    candidate = raw_content

    if has_boundary_literal_escape:
        candidate = remove_boundary_literal_escapes(raw_content)

        try:
            value = json.loads(candidate)
        except JSONDecodeError:
            pass
        else:
            return {
                "success": True,
                "value": value,
                "method": BOUNDARY_JSON_DIAGNOSTIC,
                "issue": BOUNDARY_LITERAL_ESCAPE,
            }

    try:
        value = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        pass
    else:
        if isinstance(value, dict):
            return {
                "success": True,
                "value": value,
                "method": PYTHON_LITERAL_DIAGNOSTIC,
                "issue": PYTHON_LITERAL_SYNTAX,
            }

    return {
        "success": False,
        "value": None,
        "method": None,
        "issue": UNCLASSIFIED_PARSE_FAILURE,
    }


def extract_tool_names(tools: Any) -> set[str]:
    """从 Hermes function wrapper 或直接工具结构中提取名称。"""
    if not isinstance(tools, list):
        return set()

    names: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function")
        name = (
            function.get("name")
            if isinstance(function, dict)
            else tool.get("name")
        )

        if isinstance(name, str):
            names.add(name)

    return names


def _escape_json_string_control_characters(text: str) -> str:
    """仅转义 JSON 字符串内部的原始控制字符。"""
    escaped_controls = {
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
    }
    output: list[str] = []
    in_string = False
    previous_was_escape = False

    for character in text:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue

        if previous_was_escape:
            output.append(character)
            previous_was_escape = False
        elif character == "\\":
            output.append(character)
            previous_was_escape = True
        elif character == '"':
            output.append(character)
            in_string = False
        elif ord(character) < 0x20:
            output.append(
                escaped_controls.get(
                    character,
                    f"\\u{ord(character):04x}",
                )
            )
        else:
            output.append(character)

    return "".join(output)


def recover_system_tools(sample: dict[str, Any]) -> dict[str, Any]:
    """提取 system ``<tools>``，并返回唯一、可验证的恢复候选。

    返回候选不代表已经修改样本；调用方仍须检查 ``candidate_available``。
    """
    system_texts = [
        message.get("value")
        for message in sample.get("conversations", [])
        if isinstance(message, dict)
        and message.get("from") == "system"
        and isinstance(message.get("value"), str)
    ]
    tag_count = 0
    nonempty_contents: list[str] = []
    strict_parse_errors: list[dict[str, str]] = []
    strict_parseable = False
    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(value: Any, method: str) -> None:
        if not extract_tool_names(value):
            return

        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        candidate = candidates.setdefault(
            canonical,
            {"value": value, "methods": set()},
        )
        candidate["methods"].add(method)

    for system_text in system_texts:
        for match in SYSTEM_TOOLS_PATTERN.finditer(system_text):
            tag_count += 1
            raw_content = match.group(1)
            payload = raw_content.strip()

            if not payload:
                continue

            nonempty_contents.append(raw_content)

            try:
                strict_value = json.loads(raw_content)
            except JSONDecodeError as error:
                strict_parse_errors.append(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            else:
                strict_parseable = True
                add_candidate(strict_value, "strict_system_tools_json")

            if not payload.startswith("["):
                continue

            prefix = system_text[:match.start()]

            for function_match in FUNCTION_WRAPPER_START_PATTERN.finditer(
                prefix
            ):
                stitched = (
                    "["
                    + prefix[function_match.start():]
                    + payload[1:]
                )
                diagnostic_candidates = (
                    ("stitch_split_function_wrapper", stitched),
                    (
                        "stitch_split_function_wrapper_and_escape_controls",
                        _escape_json_string_control_characters(stitched),
                    ),
                )

                for method, diagnostic_text in diagnostic_candidates:
                    try:
                        value = json.loads(diagnostic_text)
                    except JSONDecodeError:
                        continue

                    add_candidate(value, method)

    candidate_available = len(candidates) == 1
    candidate_value: Any = None
    candidate_names: list[str] = []
    methods: list[str] = []

    if candidate_available:
        candidate = next(iter(candidates.values()))
        candidate_value = candidate["value"]
        candidate_names = sorted(extract_tool_names(candidate_value))
        methods = sorted(candidate["methods"])

    return {
        "tag_count": tag_count,
        "nonempty_contents": nonempty_contents,
        "strict_parseable": strict_parseable,
        "strict_parse_errors": strict_parse_errors,
        "candidate_available": candidate_available,
        "unique_candidate_count": len(candidates),
        "candidate_value": candidate_value,
        "candidate_tool_names": candidate_names,
        "diagnostic_methods": methods,
        "_candidate_values": [
            candidate["value"] for candidate in candidates.values()
        ],
    }


def json_path_property(path: str, property_name: str) -> str:
    """生成可读 JSON path。"""
    if property_name.isidentifier():
        return f"{path}.{property_name}"

    return f"{path}[{json.dumps(property_name, ensure_ascii=False)}]"


def json_value_matches_type(value: Any, expected_type: str) -> bool:
    """按 JSON 类型语义判断值类型，避免 bool 被当成数字。"""
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None

    return False


def invalid_enum_schema_error(
    schema: dict[str, Any],
    path: str,
) -> dict[str, Any] | None:
    """返回 enum 约束自身的问题；约束可满足时返回 ``None``。"""
    if "enum" not in schema:
        return None

    enum_values = schema["enum"]
    schema_type = schema.get("type")

    if not isinstance(enum_values, list) or not enum_values:
        return {
            "schema_path": f"{path}.enum",
            "reason": "enum_must_be_nonempty_array",
            "declared_type": schema_type,
            "enum_values": enum_values,
        }

    if schema_type in JSON_SCHEMA_TYPES and not any(
        json_value_matches_type(value, schema_type)
        for value in enum_values
    ):
        return {
            "schema_path": f"{path}.enum",
            "reason": "enum_has_no_value_matching_declared_type",
            "declared_type": schema_type,
            "enum_values": enum_values,
        }

    return None


def iter_invalid_enum_nodes(
    schema: Any,
    schema_path: str = "$",
    argument_segments: tuple[tuple[str, Any], ...] = (),
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """递归返回 ``(坏 enum 记录, 所在 Schema 节点)``。"""
    if not isinstance(schema, dict):
        return

    error = invalid_enum_schema_error(schema, schema_path)

    if error is not None:
        yield (
            {
                **error,
                "argument_segments": [
                    list(item) for item in argument_segments
                ],
            },
            schema,
        )

    properties = schema.get("properties")

    if isinstance(properties, dict):
        for name, child in properties.items():
            if not isinstance(name, str):
                continue

            yield from iter_invalid_enum_nodes(
                child,
                json_path_property(f"{schema_path}.properties", name),
                (*argument_segments, ("property", name)),
            )

    if isinstance(schema.get("items"), dict):
        yield from iter_invalid_enum_nodes(
            schema["items"],
            f"{schema_path}.items",
            (*argument_segments, ("items", None)),
        )


def iter_invalid_enum_sites(
    schema: Any,
    schema_path: str = "$",
    argument_segments: tuple[tuple[str, Any], ...] = (),
) -> Iterable[dict[str, Any]]:
    """递归遍历 Schema 并返回可序列化的坏 enum 定位记录。"""
    for site, _ in iter_invalid_enum_nodes(
        schema,
        schema_path,
        argument_segments,
    ):
        yield site

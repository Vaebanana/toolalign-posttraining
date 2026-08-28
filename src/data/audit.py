"""Hermes工具调用前六层正式审计模块。

本文件审计strict JSON语法、tool call自身结构、tool name与候选tools的一致性，
调查missing_tools样本的顶层与system工具定义，检查arguments与工具Schema，
并定向调查坏enum定义。诊断候选只用于确认问题及可恢复性，不修改原始数据。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, NotRequired, TypedDict

from hermes_utils import (
    JSON_SCHEMA_TYPES,
    diagnose_tool_call_text,
    extract_tool_names,
    invalid_enum_schema_error as _invalid_enum_schema_error,
    iter_invalid_enum_sites as _iter_invalid_enum_sites,
    json_path_property as _json_path_property,
    json_value_matches_type as _json_value_matches_type,
    recover_system_tools,
)


BOUNDARY_LITERAL_ESCAPE = "boundary_literal_escape"
PYTHON_LITERAL_SYNTAX = "python_literal_syntax"
UNCLASSIFIED_PARSE_FAILURE = "unclassified_parse_failure"
INVALID_TOOL_CALL_TYPE = "invalid_tool_call_type"
NESTED_TOOL_NAME = "nested_tool_name"
MISSING_TOOL_NAME = "missing_tool_name"
MISSING_ARGUMENTS = "missing_arguments"
INVALID_ARGUMENTS_TYPE = "invalid_arguments_type"
UNKNOWN_TOOL = "unknown_tool"
MISSING_TOOLS = "missing_tools"
MISSING_SYSTEM_TOOL_DEFINITION = "missing_system_tool_definition"
MALFORMED_SYSTEM_TOOL_DEFINITION = "malformed_system_tool_definition"
INVALID_TOOL_SCHEMA = "invalid_tool_schema"
ARGUMENT_NAME_MISMATCH = "argument_name_mismatch"
MISSING_REQUIRED_ARGUMENT = "missing_required_argument"
UNEXPECTED_ARGUMENT = "unexpected_argument"
ARGUMENT_TYPE_MISMATCH = "argument_type_mismatch"
ENUM_MISMATCH = "enum_mismatch"

BOUNDARY_JSON_DIAGNOSTIC = "boundary_literal_escape_removed_json"
PYTHON_LITERAL_DIAGNOSTIC = "python_literal"

class DiagnosticParseResult(TypedDict):
    success: bool
    value: Any
    method: str | None
    issue: str


class ToolCallAudit(TypedDict):
    sample_id: Any
    message_index: int
    tool_call_index: int
    raw_content: str
    parse: dict[str, Any]
    issues: list[str]
    audit_layers_completed: list[int]
    tool_name_validation: NotRequired[ToolNameValidation]
    tool_definition_investigation: NotRequired[dict[str, Any]]
    argument_schema_validation: NotRequired[dict[str, Any]]
    invalid_tool_schema_investigation: NotRequired[dict[str, Any]]


class ToolNameValidation(TypedDict):
    candidate_name: Any
    verified: bool
    source: str | None


class ToolNameAuditResult(TypedDict):
    validation: ToolNameValidation
    issues: list[str]


def classify_failed_tool_call(raw_content: str) -> list[str]:
    """按明确优先级为一条strict parse failure分配唯一问题类型。"""
    return [diagnose_failed_tool_call(raw_content)["issue"]]


def diagnose_failed_tool_call(
    raw_content: str,
) -> DiagnosticParseResult:
    """诊断strict parse failure并返回仅供后续审计使用的candidate。"""
    return diagnose_tool_call_text(raw_content)  # type: ignore[return-value]


def audit_tool_call_structure(tool_call: Any) -> list[str]:
    """只审计tool call自身类型、name和arguments的基础结构。"""
    if not isinstance(tool_call, dict):
        return [INVALID_TOOL_CALL_TYPE]

    issues: list[str] = []
    arguments = tool_call.get("arguments")

    if "name" not in tool_call:
        if isinstance(arguments, dict) and "name" in arguments:
            issues.append(NESTED_TOOL_NAME)
        else:
            issues.append(MISSING_TOOL_NAME)

    if "arguments" not in tool_call:
        issues.append(MISSING_ARGUMENTS)
    elif not isinstance(arguments, dict):
        issues.append(INVALID_ARGUMENTS_TYPE)

    return issues


def _new_audit_record(
    tool_call: dict[str, Any],
    issues: list[str],
    audit_layers_completed: list[int],
    diagnostic_parse_method: str | None = None,
) -> ToolCallAudit:
    """根据parser记录创建问题审计记录，不写入candidate或修复值。"""
    return {
        "sample_id": tool_call["sample_id"],
        "message_index": tool_call["message_index"],
        "tool_call_index": tool_call["tool_call_index"],
        "raw_content": tool_call["raw_content"],
        "parse": {
            "strict_json": tool_call["parse_success"],
            "parsed_value": tool_call["parsed_value"],
            "diagnostic_parse_method": diagnostic_parse_method,
            "error_type": tool_call["error_type"],
            "error": tool_call["error"],
        },
        "issues": issues,
        "audit_layers_completed": audit_layers_completed,
    }


def audit_tool_call_syntax(
    tool_call: dict[str, Any],
) -> ToolCallAudit:
    """Layer 01只接受一条strict parse failure并记录语法问题。"""
    if tool_call["parse_success"]:
        raise ValueError("Audit Layer 01只接受strict parse failure")

    diagnosis = diagnose_failed_tool_call(tool_call["raw_content"])
    return _new_audit_record(
        tool_call=tool_call,
        issues=[diagnosis["issue"]],
        audit_layers_completed=[1],
        diagnostic_parse_method=diagnosis["method"],
    )


def _structure_candidate(
    tool_call: dict[str, Any],
) -> tuple[bool, Any]:
    """取得仅供Layer 02检查使用的strict value或diagnostic candidate。"""
    if tool_call["parse_success"]:
        return True, tool_call["parsed_value"]

    diagnosis = diagnose_failed_tool_call(tool_call["raw_content"])
    return diagnosis["success"], diagnosis["value"]


def audit_tool_call_structure_layer(
    tool_call: dict[str, Any],
    audit_record: ToolCallAudit | None,
) -> ToolCallAudit | None:
    """Layer 02更新已有问题记录，或为新结构问题创建记录。"""
    if audit_record is not None:
        identity = (
            tool_call["sample_id"],
            tool_call["message_index"],
            tool_call["tool_call_index"],
            tool_call["raw_content"],
        )
        record_identity = (
            audit_record["sample_id"],
            audit_record["message_index"],
            audit_record["tool_call_index"],
            audit_record["raw_content"],
        )

        if identity != record_identity:
            raise ValueError("parser记录与累积audit记录身份不一致")

        if audit_record["audit_layers_completed"] != [1]:
            raise ValueError("Layer 02已有记录必须来自Layer 01")
    elif not tool_call["parse_success"]:
        raise ValueError("strict parse failure缺少Layer 01审计记录")

    candidate_available, candidate_value = _structure_candidate(tool_call)
    structure_issues = (
        audit_tool_call_structure(candidate_value)
        if candidate_available
        else []
    )

    if audit_record is None:
        if not structure_issues:
            return None

        return _new_audit_record(
            tool_call=tool_call,
            issues=structure_issues,
            audit_layers_completed=[2],
        )

    updated_record = dict(audit_record)
    updated_record["parse"] = dict(audit_record["parse"])
    updated_record["issues"] = list(audit_record["issues"])

    updated_record["issues"].extend(structure_issues)

    updated_record["audit_layers_completed"] = [1, 2]
    return updated_record


def audit_failed_tool_call(
    tool_call: dict[str, Any],
) -> ToolCallAudit:
    """为一条parser产生的失败记录生成不可变的结构化审计结果。"""
    if tool_call["parse_success"]:
        raise ValueError("audit_failed_tool_call只接受严格解析失败记录")

    return audit_tool_call_syntax(tool_call)


def audit_failed_tool_calls(
    tool_calls: Iterable[dict[str, Any]],
) -> list[ToolCallAudit]:
    """筛选严格解析失败记录并逐条生成审计结果。"""
    return [
        audit_failed_tool_call(tool_call)
        for tool_call in tool_calls
        if not tool_call["parse_success"]
    ]


def audit_tool_calls_syntax(
    tool_calls: Iterable[dict[str, Any]],
) -> list[ToolCallAudit]:
    """执行Layer 01；输入中如有strict成功记录则拒绝运行。"""
    return [audit_tool_call_syntax(tool_call) for tool_call in tool_calls]


def audit_tool_calls_structure_layer(
    tool_calls: Iterable[dict[str, Any]],
    audit_records: Iterable[ToolCallAudit],
) -> list[ToolCallAudit]:
    """执行Layer 02；扫描全量调用并合并已有问题记录。"""
    tool_call_list = list(tool_calls)
    audit_record_list = list(audit_records)
    records_by_key: dict[tuple[Any, int], ToolCallAudit] = {}

    for record in audit_record_list:
        key = (record["sample_id"], record["tool_call_index"])

        if key in records_by_key:
            raise ValueError(f"重复的Layer 01审计记录：{key}")

        records_by_key[key] = record

    updated_records: list[ToolCallAudit] = []

    for tool_call in tool_call_list:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])
        existing_record = records_by_key.pop(key, None)
        updated_record = audit_tool_call_structure_layer(
            tool_call,
            existing_record,
        )

        if updated_record is not None:
            updated_records.append(updated_record)

    if records_by_key:
        raise ValueError(
            "Layer 01存在无法匹配到全量tool calls的记录："
            f"{sorted(records_by_key)}"
        )

    return updated_records


def audit_tool_name(tool_call: Any, tools: Any) -> ToolNameAuditResult:
    """诊断性取得调用名称，并验证它是否存在于当前样本候选tools中。"""
    candidate_name: Any = None
    source: str | None = None

    if isinstance(tool_call, dict):
        if "name" in tool_call:
            candidate_name = tool_call["name"]
            source = "top_level_name"
        else:
            arguments = tool_call.get("arguments")

            if isinstance(arguments, dict) and "name" in arguments:
                candidate_name = arguments["name"]
                source = "nested_arguments_name"

    validation: ToolNameValidation = {
        "candidate_name": candidate_name,
        "verified": False,
        "source": source,
    }

    if source is None:
        return {
            "validation": validation,
            "issues": [MISSING_TOOL_NAME],
        }

    tool_names = extract_tool_names(tools)

    if not tool_names:
        return {
            "validation": validation,
            "issues": [MISSING_TOOLS],
        }

    if candidate_name not in tool_names:
        return {
            "validation": validation,
            "issues": [UNKNOWN_TOOL],
        }

    validation["verified"] = True
    return {"validation": validation, "issues": []}


def audit_tool_calls_tool_name_layer(
    tool_calls: Iterable[dict[str, Any]],
    tools_by_sample_id: Mapping[Any, Any],
    audit_records: Iterable[ToolCallAudit],
) -> tuple[list[ToolCallAudit], list[ToolNameAuditResult]]:
    """执行Layer 03；扫描全量调用并更新或新增问题记录。"""
    audit_record_list = list(audit_records)
    records_by_key: dict[tuple[Any, int], ToolCallAudit] = {}

    for record in audit_record_list:
        key = (record["sample_id"], record["tool_call_index"])

        if key in records_by_key:
            raise ValueError(f"重复的Layer 02审计记录：{key}")

        records_by_key[key] = record

    updated_records: list[ToolCallAudit] = []
    results: list[ToolNameAuditResult] = []

    for tool_call in tool_calls:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])
        existing_record = records_by_key.pop(key, None)

        if existing_record is not None:
            completed_layers = existing_record["audit_layers_completed"]

            if completed_layers not in ([1, 2], [2]):
                raise ValueError(
                    "Layer 03已有记录必须来自Layer 02："
                    f"{existing_record}"
                )
        elif not tool_call["parse_success"]:
            raise ValueError("strict parse failure缺少前两层累积审计记录")

        candidate_available, candidate_value = _structure_candidate(
            tool_call
        )
        result = audit_tool_name(
            candidate_value if candidate_available else None,
            tools_by_sample_id.get(tool_call["sample_id"]),
        )
        validation = result["validation"]
        results.append(result)

        if existing_record is None:
            if not result["issues"]:
                continue

            new_record = _new_audit_record(
                tool_call=tool_call,
                issues=list(result["issues"]),
                audit_layers_completed=[3],
            )
            new_record["tool_name_validation"] = validation
            updated_records.append(new_record)
            continue

        updated_record = dict(existing_record)
        updated_record["parse"] = dict(existing_record["parse"])
        updated_record["issues"] = list(existing_record["issues"])

        for issue in result["issues"]:
            if issue not in updated_record["issues"]:
                updated_record["issues"].append(issue)

        updated_record["audit_layers_completed"] = [
            *existing_record["audit_layers_completed"],
            3,
        ]
        updated_record["tool_name_validation"] = validation
        updated_records.append(updated_record)

    if records_by_key:
        raise ValueError(
            "Layer 02存在无法匹配到全量tool calls的记录："
            f"{sorted(records_by_key)}"
        )

    return updated_records, results


def investigate_missing_tool_definition(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """按sample调查顶层tools与system tools，仅产生诊断候选。"""
    top_level_field_present = "tools" in sample
    raw_top_level_tools = sample.get("tools")
    top_level_parse_success = False
    top_level_parse_error_type: str | None = None
    top_level_parse_error: str | None = None
    parsed_top_level_tools: Any = None

    if not top_level_field_present:
        top_level_state = "missing"
    elif isinstance(raw_top_level_tools, str):
        try:
            parsed_top_level_tools = json.loads(raw_top_level_tools)
        except JSONDecodeError as error:
            top_level_state = "parse_failed"
            top_level_parse_error_type = type(error).__name__
            top_level_parse_error = str(error)
        else:
            top_level_parse_success = True
            top_level_state = (
                "empty"
                if isinstance(parsed_top_level_tools, list)
                and not parsed_top_level_tools
                else "present"
            )
    else:
        parsed_top_level_tools = raw_top_level_tools
        top_level_state = (
            "empty"
            if isinstance(parsed_top_level_tools, list)
            and not parsed_top_level_tools
            else "present_unparsed_type"
        )

    system_recovery = recover_system_tools(sample)
    tag_count = system_recovery["tag_count"]
    nonempty_contents = system_recovery["nonempty_contents"]
    strict_parse_errors = system_recovery["strict_parse_errors"]
    strict_parseable = system_recovery["strict_parseable"]
    nonempty_tag_count = len(nonempty_contents)
    system_tools_present = nonempty_tag_count > 0
    system_tools_malformed = (
        system_tools_present and not strict_parseable
    )
    recovery_candidate_available = system_recovery["candidate_available"]
    recovered_tool_names = system_recovery["candidate_tool_names"]
    recovery_methods = system_recovery["diagnostic_methods"]

    issues: list[str] = []

    if not system_tools_present:
        issues.append(MISSING_SYSTEM_TOOL_DEFINITION)
    elif system_tools_malformed:
        issues.append(MALFORMED_SYSTEM_TOOL_DEFINITION)

    return {
        "sample_id": sample.get("id"),
        "top_level_tools": {
            "field_present": top_level_field_present,
            "raw_value": raw_top_level_tools,
            "state": top_level_state,
            "strict_parse_success": top_level_parse_success,
            "error_type": top_level_parse_error_type,
            "error": top_level_parse_error,
        },
        "system_tools": {
            "tag_count": tag_count,
            "nonempty_tag_count": nonempty_tag_count,
            "present": system_tools_present,
            "raw_contents": nonempty_contents,
            "strict_parseable": strict_parseable,
            "malformed": system_tools_malformed,
            "strict_parse_errors": strict_parse_errors,
        },
        "recovery": {
            "candidate_available": recovery_candidate_available,
            "unique_candidate_count": system_recovery[
                "unique_candidate_count"
            ],
            "candidate_tool_names": recovered_tool_names,
            "diagnostic_methods": recovery_methods,
        },
        # 只供后续审计层使用，不会写入Layer 04或累计审计文件。
        "_diagnostic_candidate_values": [
            candidate
            for candidate in system_recovery["_candidate_values"]
        ],
        "issues": issues,
    }


def audit_missing_tool_definitions_layer(
    tool_calls: Iterable[dict[str, Any]],
    raw_samples_by_id: Mapping[Any, dict[str, Any]],
    audit_records: Iterable[ToolCallAudit],
) -> tuple[list[ToolCallAudit], list[dict[str, Any]]]:
    """执行Layer 04；只调查带missing_tools的目标调用及其sample。"""
    records_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in audit_records
    }
    investigations_by_sample: dict[Any, dict[str, Any]] = {}
    updated_records: list[ToolCallAudit] = []
    seen_call_keys: set[tuple[Any, int]] = set()

    for tool_call in tool_calls:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])

        if key in seen_call_keys:
            raise ValueError(f"重复的Layer 04目标调用：{key}")

        seen_call_keys.add(key)
        existing_record = records_by_key.get(key)

        if existing_record is None or MISSING_TOOLS not in existing_record["issues"]:
            raise ValueError(f"Layer 04目标不是missing_tools问题记录：{key}")

        sample_id = tool_call["sample_id"]

        if sample_id not in raw_samples_by_id:
            raise ValueError(f"找不到Layer 04原始sample：{sample_id!r}")

        if sample_id not in investigations_by_sample:
            investigations_by_sample[sample_id] = (
                investigate_missing_tool_definition(
                    raw_samples_by_id[sample_id]
                )
            )

        investigation = investigations_by_sample[sample_id]
        updated_record = dict(existing_record)
        updated_record["parse"] = dict(existing_record["parse"])
        updated_record["issues"] = list(existing_record["issues"])

        for issue in investigation["issues"]:
            if issue not in updated_record["issues"]:
                updated_record["issues"].append(issue)

        completed_layers = existing_record["audit_layers_completed"]

        if not completed_layers or completed_layers[-1] != 3:
            raise ValueError("Layer 04已有记录必须完成Layer 03")

        candidate_name = existing_record[
            "tool_name_validation"
        ]["candidate_name"]
        recovery = dict(investigation["recovery"])
        recovery["candidate_tool_name"] = candidate_name
        recovery["candidate_available"] = (
            investigation["recovery"]["candidate_available"]
            and candidate_name
            in investigation["recovery"]["candidate_tool_names"]
        )
        record_investigation = {
            "top_level_tools": dict(investigation["top_level_tools"]),
            "system_tools": dict(investigation["system_tools"]),
            "tool_definition_recovery": recovery,
        }
        updated_record["tool_definition_investigation"] = (
            record_investigation
        )
        updated_record["audit_layers_completed"] = [
            *completed_layers,
            4,
        ]
        updated_records.append(updated_record)

    return updated_records, list(investigations_by_sample.values())


def _json_value_type(value: Any) -> str:
    """返回适合审计记录的JSON类型名称。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"

    return type(value).__name__


def _argument_name_tokens(name: str) -> set[str]:
    """提取参数名语义词元，并归一query/question等明确同义形式。"""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    raw_tokens = re.findall(r"[A-Za-z0-9]+", expanded.lower())
    aliases = {
        "queries": "question",
        "query": "question",
        "questions": "question",
        "question": "question",
    }
    return {aliases.get(token, token) for token in raw_tokens}


def _validate_schema_structure(
    schema: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    """验证本数据使用的JSON Schema核心结构，不校验arguments。"""
    errors: list[dict[str, Any]] = []

    if not isinstance(schema, dict):
        return [
            {
                "schema_path": path,
                "reason": "schema_must_be_object",
                "actual_type": _json_value_type(schema),
            }
        ]

    schema_type = schema.get("type")

    if "type" in schema and schema_type not in JSON_SCHEMA_TYPES:
        errors.append(
            {
                "schema_path": f"{path}.type",
                "reason": "invalid_type",
                "value": schema_type,
            }
        )

    properties = schema.get("properties")

    if properties is not None:
        if not isinstance(properties, dict):
            errors.append(
                {
                    "schema_path": f"{path}.properties",
                    "reason": "properties_must_be_object",
                    "actual_type": _json_value_type(properties),
                }
            )
        else:
            for property_name, property_schema in properties.items():
                if not isinstance(property_name, str):
                    errors.append(
                        {
                            "schema_path": f"{path}.properties",
                            "reason": "property_name_must_be_string",
                        }
                    )
                    continue

                errors.extend(
                    _validate_schema_structure(
                        property_schema,
                        _json_path_property(
                            f"{path}.properties",
                            property_name,
                        ),
                    )
                )

    required = schema.get("required")

    if required is not None and (
        not isinstance(required, list)
        or not all(isinstance(name, str) for name in required)
        or len(required) != len(set(required))
    ):
        errors.append(
            {
                "schema_path": f"{path}.required",
                "reason": "required_must_be_unique_string_array",
            }
        )

    if "items" in schema:
        errors.extend(
            _validate_schema_structure(
                schema["items"],
                f"{path}.items",
            )
        )

    enum_error = _invalid_enum_schema_error(schema, path)

    if enum_error is not None:
        errors.append(enum_error)

    additional_properties = schema.get("additionalProperties")

    if additional_properties is not None and not isinstance(
        additional_properties,
        (bool, dict),
    ):
        errors.append(
            {
                "schema_path": f"{path}.additionalProperties",
                "reason": "additional_properties_must_be_boolean_or_schema",
            }
        )
    elif isinstance(additional_properties, dict):
        errors.extend(
            _validate_schema_structure(
                additional_properties,
                f"{path}.additionalProperties",
            )
        )

    return errors


def _validate_value_against_schema(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[dict[str, Any]]:
    """递归检查类型、对象参数名、required、array items和enum。"""
    details: list[dict[str, Any]] = []
    if not schema:
        return []

    expected_type = schema.get("type")

    if (
        expected_type is not None
        and not _json_value_matches_type(value, expected_type)
    ):
        return [
            {
                "issue": ARGUMENT_TYPE_MISMATCH,
                "argument_path": path,
                "expected_type": expected_type,
                "actual_type": _json_value_type(value),
            }
        ]

    enum_values = schema.get("enum")
    enum_is_usable = (
        isinstance(enum_values, list)
        and bool(enum_values)
        and (
            expected_type not in JSON_SCHEMA_TYPES
            or any(
                _json_value_matches_type(enum_value, expected_type)
                for enum_value in enum_values
            )
        )
    )

    if enum_is_usable and value not in enum_values:
        details.append(
            {
                "issue": ENUM_MISMATCH,
                "argument_path": path,
                "value": value,
                "allowed_values": enum_values,
            }
        )

    if expected_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        unexpected_names = [
            name for name in value if name not in properties
        ]
        missing_names = [
            name for name in required if name not in value
        ]

        for name in unexpected_names:
            details.append(
                {
                    "issue": UNEXPECTED_ARGUMENT,
                    "argument_path": _json_path_property(path, name),
                    "argument_name": name,
                }
            )

        for name in missing_names:
            details.append(
                {
                    "issue": MISSING_REQUIRED_ARGUMENT,
                    "argument_path": _json_path_property(path, name),
                    "argument_name": name,
                }
            )

        # 只有一个实际参数、一个Schema参数、一个缺失required参数，且类型
        # 一致时，才把纯Schema问题进一步标记为唯一参数名映射候选。
        if (
            len(value) == 1
            and len(properties) == 1
            and len(unexpected_names) == 1
            and len(missing_names) == 1
        ):
            provided_name = unexpected_names[0]
            expected_name = missing_names[0]
            expected_schema = properties.get(expected_name)
            semantic_overlap = sorted(
                _argument_name_tokens(provided_name)
                & _argument_name_tokens(expected_name)
            )

            if (
                isinstance(expected_schema, dict)
                and expected_schema.get("type") in JSON_SCHEMA_TYPES
                and semantic_overlap
                and _json_value_matches_type(
                    value[provided_name],
                    expected_schema["type"],
                )
            ):
                details.append(
                    {
                        "issue": ARGUMENT_NAME_MISMATCH,
                        "object_path": path,
                        "provided_argument": provided_name,
                        "schema_argument_candidate": expected_name,
                        "evidence": {
                            "unique_provided_argument": True,
                            "unique_schema_property": True,
                            "missing_required_candidate": True,
                            "matching_json_type": expected_schema["type"],
                            "semantic_name_token_overlap": (
                                semantic_overlap
                            ),
                        },
                    }
                )

        for name in value.keys() & properties.keys():
            details.extend(
                _validate_value_against_schema(
                    value[name],
                    properties[name],
                    _json_path_property(path, name),
                )
            )

    elif expected_type == "array" and "items" in schema:
        item_schema = schema["items"]

        for index, item in enumerate(value):
            details.extend(
                _validate_value_against_schema(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                )
            )

    return details


def _tool_definitions_by_name(tools: Any) -> dict[str, list[dict[str, Any]]]:
    """保留同名定义，以便拒绝存在冲突的候选Schema。"""
    definitions: dict[str, list[dict[str, Any]]] = {}

    if not isinstance(tools, list):
        return definitions

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        definition = tool.get("function", tool)

        if not isinstance(definition, dict):
            continue

        name = definition.get("name")

        if isinstance(name, str):
            definitions.setdefault(name, []).append(definition)

    return definitions


def audit_argument_schema(
    tool_call: Any,
    top_level_tools: Any,
    diagnostic_tools: Any = None,
) -> dict[str, Any]:
    """审计一条调用的arguments与其唯一候选工具Schema是否一致。"""
    call_name: Any = None
    arguments: Any = None
    call_name_source: str | None = None
    argument_projection_method: str | None = None

    if isinstance(tool_call, dict):
        arguments = tool_call.get("arguments")

        if isinstance(tool_call.get("name"), str):
            call_name = tool_call["name"]
            call_name_source = "top_level_name"
        elif (
            isinstance(arguments, dict)
            and isinstance(arguments.get("name"), str)
        ):
            call_name = arguments["name"]
            call_name_source = "nested_arguments_name"
            arguments = {
                name: value
                for name, value in arguments.items()
                if name != "name"
            }
            argument_projection_method = (
                "exclude_nested_tool_name_for_argument_audit"
            )

    top_level_definitions = _tool_definitions_by_name(top_level_tools)
    diagnostic_definitions = _tool_definitions_by_name(diagnostic_tools)

    if call_name in top_level_definitions:
        definitions = top_level_definitions[call_name]
        definition_source = "top_level_tools"
    elif call_name in diagnostic_definitions:
        definitions = diagnostic_definitions[call_name]
        definition_source = "diagnostic_system_tools"
    else:
        definitions = []
        definition_source = None

    base_result = {
        "schema_checked": False,
        "tool_name": call_name,
        "tool_name_source": call_name_source,
        "tool_definition_source": definition_source,
        "argument_projection_method": argument_projection_method,
        "valid_against_schema": False,
        "issues": [],
        "issue_details": [],
    }

    if not definitions:
        base_result["not_checked_reason"] = "tool_definition_unavailable"
        return base_result

    canonical_definitions = {
        json.dumps(
            definition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ): definition
        for definition in definitions
    }

    if len(canonical_definitions) != 1:
        base_result["issues"] = [INVALID_TOOL_SCHEMA]
        base_result["issue_details"] = [
            {
                "issue": INVALID_TOOL_SCHEMA,
                "reason": "conflicting_tool_definitions",
                "definition_count": len(canonical_definitions),
            }
        ]
        return base_result

    definition = next(iter(canonical_definitions.values()))
    schema = definition.get("parameters")
    schema_errors = _validate_schema_structure(schema)

    if isinstance(schema, dict) and schema.get("type") != "object":
        schema_errors.append(
            {
                "schema_path": "$.type",
                "reason": "tool_parameters_root_must_be_object",
                "value": schema.get("type"),
            }
        )

    if schema_errors:
        invalid_schema_detail = {
            "issue": INVALID_TOOL_SCHEMA,
            "schema_errors": schema_errors,
        }
        issue_details = [invalid_schema_detail]
        enum_schema_error_reasons = {
            "enum_must_be_nonempty_array",
            "enum_has_no_value_matching_declared_type",
        }

        # enum约束自身损坏时，跳过该enum约束，但仍可诊断性检查同一Schema
        # 中不受影响的required和type，以保留原始问题的完整记录。
        if (
            isinstance(schema, dict)
            and schema.get("type") == "object"
            and all(
                error["reason"] in enum_schema_error_reasons
                for error in schema_errors
            )
        ):
            issue_details.extend(
                _validate_value_against_schema(
                    arguments,
                    schema,
                    "$.arguments",
                )
            )
            base_result["argument_validation_mode"] = (
                "partial_excluding_invalid_enum_constraints"
            )

        base_result["issues"] = list(
            dict.fromkeys(
                detail["issue"] for detail in issue_details
            )
        )
        base_result["issue_details"] = issue_details
        base_result["not_checked_reason"] = INVALID_TOOL_SCHEMA
        return base_result

    validation_details = _validate_value_against_schema(
        arguments,
        schema,
        "$.arguments",
    )
    issue_names = list(
        dict.fromkeys(
            detail["issue"] for detail in validation_details
        )
    )
    base_result["schema_checked"] = True
    base_result["issues"] = issue_names
    base_result["issue_details"] = validation_details
    base_result["valid_against_schema"] = not issue_names
    return base_result


def audit_tool_calls_argument_schema_layer(
    tool_calls: Iterable[dict[str, Any]],
    tools_by_sample_id: Mapping[Any, Any],
    diagnostic_tools_by_sample_id: Mapping[Any, Any],
    audit_records: Iterable[ToolCallAudit],
) -> tuple[list[ToolCallAudit], list[dict[str, Any]]]:
    """执行Layer 05；扫描全量调用并累计arguments/Schema问题。"""
    records_by_key: dict[tuple[Any, int], ToolCallAudit] = {}

    for record in audit_records:
        key = (record["sample_id"], record["tool_call_index"])

        if key in records_by_key:
            raise ValueError(f"重复的Layer 04累计审计记录：{key}")

        records_by_key[key] = record

    updated_records: list[ToolCallAudit] = []
    results: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])
        existing_record = records_by_key.pop(key, None)
        candidate_available, candidate_value = _structure_candidate(
            tool_call
        )
        result = audit_argument_schema(
            candidate_value if candidate_available else None,
            tools_by_sample_id.get(tool_call["sample_id"]),
            diagnostic_tools_by_sample_id.get(tool_call["sample_id"]),
        )
        results.append(result)

        if existing_record is None:
            if not result["issues"]:
                continue

            new_record = _new_audit_record(
                tool_call=tool_call,
                issues=list(result["issues"]),
                audit_layers_completed=[5],
            )
            new_record["argument_schema_validation"] = result
            updated_records.append(new_record)
            continue

        completed_layers = existing_record["audit_layers_completed"]

        if not completed_layers or completed_layers[-1] not in (3, 4):
            raise ValueError(
                "Layer 05已有记录必须完成Layer 03或Layer 04"
            )

        updated_record = dict(existing_record)
        updated_record["parse"] = dict(existing_record["parse"])
        updated_record["issues"] = list(existing_record["issues"])

        for issue in result["issues"]:
            if issue not in updated_record["issues"]:
                updated_record["issues"].append(issue)

        updated_record["audit_layers_completed"] = [
            *completed_layers,
            5,
        ]
        updated_record["argument_schema_validation"] = result
        updated_records.append(updated_record)

    if records_by_key:
        raise ValueError(
            "Layer 04累计记录存在无法匹配到全量tool calls的记录："
            f"{sorted(records_by_key)}"
        )

    return updated_records, results


def _values_at_argument_segments(
    arguments: Any,
    segments: Iterable[Iterable[Any]],
) -> list[Any]:
    """从一条arguments中取得某Schema节点对应的全部观测值。"""
    values = [arguments]

    for segment in segments:
        segment_kind, segment_value = segment
        next_values: list[Any] = []

        if segment_kind == "property":
            for value in values:
                if isinstance(value, dict) and segment_value in value:
                    next_values.append(value[segment_value])
        elif segment_kind == "items":
            for value in values:
                if isinstance(value, list):
                    next_values.extend(value)
        else:
            raise ValueError(f"未知的argument segment：{segment_kind!r}")

        values = next_values

    return values


def _diagnostic_call_name_and_arguments(
    tool_call: dict[str, Any],
) -> tuple[Any, Any]:
    """取得仅供审计使用的调用名称与参数投影。"""
    candidate_available, candidate_value = _structure_candidate(tool_call)

    if not candidate_available or not isinstance(candidate_value, dict):
        return None, None

    arguments = candidate_value.get("arguments")
    call_name = candidate_value.get("name")

    if (
        not isinstance(call_name, str)
        and isinstance(arguments, dict)
        and isinstance(arguments.get("name"), str)
    ):
        call_name = arguments["name"]
        arguments = {
            name: value
            for name, value in arguments.items()
            if name != "name"
        }

    return call_name, arguments


def investigate_invalid_tool_schema_definition(
    definition: dict[str, Any],
    call_arguments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """按唯一定义调查坏enum模式及删除enum候选的证据。"""
    call_argument_list = list(call_arguments)
    schema = definition.get("parameters")
    schema_sites: list[dict[str, Any]] = []

    for invalid_site in _iter_invalid_enum_sites(schema):
        observed_by_call: list[dict[str, Any]] = []
        observed_values: list[Any] = []

        for call in call_argument_list:
            values = _values_at_argument_segments(
                call["arguments"],
                invalid_site["argument_segments"],
            )

            if values:
                observed_by_call.append(
                    {
                        "tool_call_index": call["tool_call_index"],
                        "values": values,
                    }
                )
                observed_values.extend(values)

        declared_type = invalid_site.get("declared_type")
        compatible_values = [
            value
            for value in observed_values
            if declared_type in JSON_SCHEMA_TYPES
            and _json_value_matches_type(value, declared_type)
        ]
        incompatible_values = [
            value
            for value in observed_values
            if declared_type not in JSON_SCHEMA_TYPES
            or not _json_value_matches_type(value, declared_type)
        ]
        enum_values = invalid_site.get("enum_values")

        if enum_values == [None] and declared_type != "null":
            placeholder_class = "singleton_null_enum_placeholder"
        elif enum_values == []:
            placeholder_class = "empty_enum_placeholder"
        else:
            placeholder_class = "meaningful_or_mixed_invalid_enum"

        if compatible_values and not incompatible_values:
            observation_evidence = "only_declared_type_compatible_values"
        elif compatible_values and incompatible_values:
            observation_evidence = "mixed_type_observations"
        elif incompatible_values:
            observation_evidence = "only_type_incompatible_values"
        else:
            observation_evidence = "not_observed"

        remove_enum_candidate = placeholder_class in {
            "singleton_null_enum_placeholder",
            "empty_enum_placeholder",
        }
        schema_sites.append(
            {
                "schema_path": invalid_site["schema_path"],
                "reason": invalid_site["reason"],
                "declared_type": declared_type,
                "raw_enum_values": enum_values,
                "placeholder_class": placeholder_class,
                "observations": observed_by_call,
                "observation_evidence": observation_evidence,
                "observed_value_count": len(observed_values),
                "declared_type_compatible_value_count": len(
                    compatible_values
                ),
                "declared_type_incompatible_value_count": len(
                    incompatible_values
                ),
                "deterministic_repair_assessment": {
                    "candidate_operation": (
                        "remove_enum" if remove_enum_candidate else None
                    ),
                    "candidate_available": remove_enum_candidate,
                    "preserves_declared_type": remove_enum_candidate,
                    "invents_enum_values": False,
                },
            }
        )

    return {
        "tool_name": definition.get("name"),
        "raw_tool_definition": definition,
        "invalid_enum_sites": schema_sites,
        "all_sites_match_placeholder_pattern": bool(schema_sites)
        and all(
            site["deterministic_repair_assessment"][
                "candidate_available"
            ]
            for site in schema_sites
        ),
    }


def audit_invalid_tool_schemas_layer(
    tool_calls: Iterable[dict[str, Any]],
    tools_by_sample_id: Mapping[Any, Any],
    diagnostic_tools_by_sample_id: Mapping[Any, Any],
    audit_records: Iterable[ToolCallAudit],
) -> tuple[list[ToolCallAudit], list[dict[str, Any]]]:
    """执行Layer 06；只调查Layer 05标记的invalid_tool_schema调用。"""
    target_calls = list(tool_calls)
    records_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in audit_records
    }
    calls_by_definition: dict[
        tuple[Any, str], list[dict[str, Any]]
    ] = {}
    target_records: dict[tuple[Any, int], ToolCallAudit] = {}

    for tool_call in target_calls:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])
        existing_record = records_by_key.get(key)

        if (
            existing_record is None
            or INVALID_TOOL_SCHEMA not in existing_record["issues"]
        ):
            raise ValueError(f"Layer 06目标不是invalid_tool_schema记录：{key}")

        if existing_record["audit_layers_completed"][-1] != 5:
            raise ValueError("Layer 06目标记录必须完成Layer 05")

        call_name, arguments = _diagnostic_call_name_and_arguments(
            tool_call
        )

        if not isinstance(call_name, str) or not isinstance(arguments, dict):
            raise ValueError(f"Layer 06无法取得调用名称或arguments：{key}")

        definition_key = (tool_call["sample_id"], call_name)
        calls_by_definition.setdefault(definition_key, []).append(
            {
                "tool_call_index": tool_call["tool_call_index"],
                "arguments": arguments,
            }
        )
        target_records[key] = existing_record

    definition_investigations: list[dict[str, Any]] = []
    investigations_by_definition: dict[
        tuple[Any, str], dict[str, Any]
    ] = {}

    for definition_key, definition_calls in calls_by_definition.items():
        sample_id, call_name = definition_key
        top_level_definitions = _tool_definitions_by_name(
            tools_by_sample_id.get(sample_id)
        )
        diagnostic_definitions = _tool_definitions_by_name(
            diagnostic_tools_by_sample_id.get(sample_id)
        )

        if call_name in top_level_definitions:
            definitions = top_level_definitions[call_name]
            definition_source = "top_level_tools"
        else:
            definitions = diagnostic_definitions.get(call_name, [])
            definition_source = "diagnostic_system_tools"

        canonical_definitions = {
            json.dumps(
                definition,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ): definition
            for definition in definitions
        }

        if len(canonical_definitions) != 1:
            raise ValueError(
                "Layer 06无法取得唯一工具定义："
                f"sample={sample_id!r}, tool={call_name!r}"
            )

        definition = next(iter(canonical_definitions.values()))
        investigation = investigate_invalid_tool_schema_definition(
            definition,
            definition_calls,
        )

        if not investigation["invalid_enum_sites"]:
            raise ValueError("Layer 06目标定义没有坏enum字段")

        definition_record = {
            "sample_id": sample_id,
            "tool_name": call_name,
            "tool_definition_source": definition_source,
            "affected_tool_call_indexes": sorted(
                call["tool_call_index"] for call in definition_calls
            ),
            **investigation,
        }
        investigations_by_definition[definition_key] = definition_record
        definition_investigations.append(definition_record)

    updated_records: list[ToolCallAudit] = []

    for tool_call in target_calls:
        key = (tool_call["sample_id"], tool_call["tool_call_index"])
        existing_record = target_records[key]
        call_name = existing_record["argument_schema_validation"][
            "tool_name"
        ]
        definition_investigation = investigations_by_definition[
            (tool_call["sample_id"], call_name)
        ]
        updated_record = dict(existing_record)
        updated_record["parse"] = dict(existing_record["parse"])
        updated_record["issues"] = list(existing_record["issues"])
        updated_record["invalid_tool_schema_investigation"] = {
            "tool_name": call_name,
            "tool_definition_source": definition_investigation[
                "tool_definition_source"
            ],
            "all_sites_match_placeholder_pattern": (
                definition_investigation[
                    "all_sites_match_placeholder_pattern"
                ]
            ),
            "invalid_enum_sites": definition_investigation[
                "invalid_enum_sites"
            ],
        }
        updated_record["audit_layers_completed"] = [
            *existing_record["audit_layers_completed"],
            6,
        ]
        updated_records.append(updated_record)

    return updated_records, definition_investigations

"""Hermes Day 01 第一阶段：基于既有审计证据的确定性修复。

本模块只实现五类已确认修复：boundary literal、Python literal、nested tool
name、空 tools 恢复和坏 enum placeholder 删除。它不做参数语义映射、格式
Normalize，也不写入完整 repaired dataset。
"""

from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

from hermes_utils import (
    extract_tool_names,
    iter_invalid_enum_nodes,
    recover_system_tools,
    remove_boundary_literal_escapes,
)
from parsers import TOOL_CALL_PATTERN


BOUNDARY_LITERAL_ESCAPE = "boundary_literal_escape"
PYTHON_LITERAL_SYNTAX = "python_literal_syntax"
NESTED_TOOL_NAME = "nested_tool_name"
MISSING_TOOLS = "missing_tools"
MALFORMED_SYSTEM_TOOL_DEFINITION = (
    "malformed_system_tool_definition"
)
INVALID_TOOL_SCHEMA = "invalid_tool_schema"

REMOVE_BOUNDARY_LITERAL_ESCAPE = "remove_boundary_literal_escape"
PYTHON_LITERAL_TO_OBJECT = "python_literal_to_object"
MOVE_NESTED_TOOL_NAME = "move_nested_tool_name"
RECOVER_SYSTEM_TOOL_DEFINITION = "recover_system_tool_definition"
REMOVE_INVALID_ENUM_PLACEHOLDER = "remove_invalid_enum_placeholder"

CallKey = tuple[Any, int]
DefinitionKey = tuple[Any, str]


@dataclass
class EnumRepairEvidence:
    """一个工具定义中允许删除的 enum site。"""

    paths: dict[str, Any] = field(default_factory=dict)
    affected_tool_call_indexes: set[int] = field(default_factory=set)


@dataclass
class RepairAuditContext:
    """从 Layer 01/03/04/06 审计产物加载的修复白名单。"""

    boundary_calls: set[CallKey] = field(default_factory=set)
    python_literal_calls: set[CallKey] = field(default_factory=set)
    verified_nested_names: dict[CallKey, str] = field(
        default_factory=dict
    )
    missing_tools: dict[Any, set[str]] = field(default_factory=dict)
    invalid_enums: dict[DefinitionKey, EnumRepairEvidence] = field(
        default_factory=dict
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"{path}:{line_number} 必须是 JSON object"
                )
            records.append(value)

    return records


def _call_key(record: dict[str, Any]) -> CallKey:
    return record["sample_id"], record["tool_call_index"]


def _set_verified_name(
    context: RepairAuditContext,
    key: CallKey,
    name: Any,
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"nested name 缺少有效审计值：{key}")

    existing = context.verified_nested_names.get(key)
    if existing is not None and existing != name:
        raise ValueError(
            f"nested name 审计证据冲突：{key}: {existing!r} != {name!r}"
        )

    context.verified_nested_names[key] = name


def load_repair_audit_context(audit_dir: str | Path) -> RepairAuditContext:
    """加载既有审计产物，构造 deterministic repair 白名单。"""
    directory = Path(audit_dir)
    context = RepairAuditContext()

    layer_01 = _read_jsonl(
        directory / "hermes_audit_layer_01_json_syntax_records.jsonl"
    )
    layer_03 = _read_jsonl(
        directory
        / "hermes_audit_layer_03_tool_name_resolution_records.jsonl"
    )
    layer_04 = _read_jsonl(
        directory
        / "hermes_audit_layer_04_missing_tool_definition_records.jsonl"
    )
    layer_06_definitions = _read_jsonl(
        directory
        / "hermes_audit_layer_06_invalid_tool_schema_definitions.jsonl"
    )

    for record in layer_01:
        key = _call_key(record)
        issues = record.get("issues", [])

        if BOUNDARY_LITERAL_ESCAPE in issues:
            context.boundary_calls.add(key)
        elif PYTHON_LITERAL_SYNTAX in issues:
            context.python_literal_calls.add(key)

    for record in layer_03:
        if NESTED_TOOL_NAME not in record.get("issues", []):
            continue

        validation = record.get("tool_name_validation", {})
        if validation.get("verified"):
            _set_verified_name(
                context,
                _call_key(record),
                validation.get("candidate_name"),
            )

    for record in layer_04:
        investigation = record.get("tool_definition_investigation", {})
        recovery = investigation.get("tool_definition_recovery", {})
        key = _call_key(record)
        name = recovery.get("candidate_tool_name")

        if (
            MISSING_TOOLS not in record.get("issues", [])
            or not recovery.get("candidate_available")
        ):
            continue

        _set_verified_name(context, key, name)
        context.missing_tools.setdefault(
            record["sample_id"], set()
        ).add(name)

    for record in layer_06_definitions:
        sample_id = record["sample_id"]
        tool_name = record["tool_name"]
        key = (sample_id, tool_name)
        evidence = context.invalid_enums.setdefault(
            key,
            EnumRepairEvidence(),
        )
        evidence.affected_tool_call_indexes.update(
            record.get("affected_tool_call_indexes", [])
        )

        for site in record.get("invalid_enum_sites", []):
            assessment = site.get(
                "deterministic_repair_assessment", {}
            )
            if (
                assessment.get("candidate_operation") != "remove_enum"
                or not assessment.get("candidate_available")
                or assessment.get("invents_enum_values") is not False
            ):
                raise ValueError(
                    "Layer 06 包含非确定性 enum 修复候选："
                    f"{sample_id!r}/{tool_name!r}"
                )

            path = site["schema_path"]
            raw_value = site.get("raw_enum_values")
            sentinel = object()
            existing = evidence.paths.get(path, sentinel)

            if existing is not sentinel and existing != raw_value:
                raise ValueError(
                    f"enum 审计证据冲突：{key}/{path}"
                )

            evidence.paths[path] = raw_value

    overlap = context.boundary_calls & context.python_literal_calls
    if overlap:
        raise ValueError(f"Layer 01 修复分类不唯一：{sorted(overlap)}")

    return context


def repair_boundary_literal_escape(text: str) -> dict[str, Any]:
    """只移除边界字面量 ``\\n``，且要求随后严格 JSON 解析为 dict。"""
    candidate = remove_boundary_literal_escapes(text)
    value = json.loads(candidate)

    if not isinstance(value, dict):
        raise TypeError(
            "boundary literal 修复后 tool_call 必须是 object"
        )

    return value


def repair_python_literal(text: str) -> dict[str, Any]:
    """安全地将经审计确认的 Python literal 解析为 dict。"""
    candidate = remove_boundary_literal_escapes(text)
    value = ast.literal_eval(candidate)

    if not isinstance(value, dict):
        raise TypeError(
            "Python literal 修复后 tool_call 必须是 dict"
        )

    return value


def repair_nested_tool_name(
    tool_call: dict[str, Any],
    verified_tool_name: str,
    available_tool_names: set[str],
) -> None:
    """仅在 nested name 与审计值、候选工具表都一致时移动 name。"""
    arguments = tool_call.get("arguments")

    if "name" in tool_call:
        raise ValueError("nested_tool_name 修复前顶层 name 必须缺失")
    if not isinstance(arguments, dict):
        raise TypeError("nested_tool_name 的 arguments 必须是 dict")

    nested_name = arguments.get("name")
    if nested_name != verified_tool_name:
        raise ValueError(
            "nested name 与审计值不一致："
            f"{nested_name!r} != {verified_tool_name!r}"
        )
    if verified_tool_name not in available_tool_names:
        raise ValueError(
            f"审计工具名不在 repaired tools 中：{verified_tool_name!r}"
        )

    tool_call["name"] = verified_tool_name
    del arguments["name"]


def repair_missing_tools(
    sample: dict[str, Any],
    expected_tool_names: set[str],
) -> list[Any]:
    """用共享 recovery primitive 恢复经 Layer 04 确认的空 tools。"""
    recovery = recover_system_tools(sample)

    if not recovery["candidate_available"]:
        raise ValueError("system tools 不再具有唯一恢复候选")

    recovered = recovery["candidate_value"]
    if not isinstance(recovered, list) or not recovered:
        raise TypeError("system tools 恢复候选必须是非空 list")

    recovered_names = extract_tool_names(recovered)
    if not expected_tool_names.issubset(recovered_names):
        raise ValueError(
            "恢复后的 tools 不包含审计工具名："
            f"{sorted(expected_tool_names - recovered_names)}"
        )

    return copy.deepcopy(recovered)


def _tool_definition(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None

    function = tool.get("function")
    if isinstance(function, dict):
        return function

    return tool


def repair_invalid_enum_placeholders(
    tools: list[Any],
    evidence_by_tool_name: dict[str, EnumRepairEvidence],
) -> tuple[int, set[int]]:
    """只删除 Layer 06 白名单中的 ``[None]`` / ``[]`` enum site。"""
    removed = 0
    affected_calls: set[int] = set()

    for tool_name, evidence in evidence_by_tool_name.items():
        definitions = [
            definition
            for tool in tools
            if (definition := _tool_definition(tool)) is not None
            and definition.get("name") == tool_name
        ]

        if len(definitions) != 1:
            raise ValueError(
                f"{tool_name!r} 应唯一对应一个工具定义，实际 {len(definitions)}"
            )

        schema = definitions[0].get("parameters")
        invalid_nodes = {
            site["schema_path"]: (site, node)
            for site, node in iter_invalid_enum_nodes(schema)
        }
        expected_paths = set(evidence.paths)
        missing_paths = expected_paths - set(invalid_nodes)

        if missing_paths:
            raise ValueError(
                f"{tool_name!r} 缺少审计 enum site：{sorted(missing_paths)}"
            )

        for path in sorted(expected_paths):
            site, node = invalid_nodes[path]
            expected_value = evidence.paths[path]
            actual_value = site.get("enum_values")
            declared_type = site.get("declared_type")

            if actual_value != expected_value:
                raise ValueError(
                    f"{tool_name!r}/{path} enum 值与审计证据不一致"
                )
            if not (
                actual_value == []
                or (actual_value == [None] and declared_type != "null")
            ):
                raise ValueError(
                    f"{tool_name!r}/{path} 不是允许删除的 placeholder"
                )

            del node["enum"]
            removed += 1

        affected_calls.update(evidence.affected_tool_call_indexes)

    return removed, affected_calls


def _parse_tools(raw_tools: Any) -> list[Any]:
    value = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools

    if not isinstance(value, list):
        raise TypeError("Hermes tools 解析结果必须是 list")

    return copy.deepcopy(value)


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def repair_hermes_sample(
    sample: dict[str, Any],
    audit_context: RepairAuditContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """修复一条副本，并返回 sample-level repair record。"""
    repaired_sample = copy.deepcopy(sample)
    sample_id = sample.get("id")
    issues: list[str] = []
    repairs: list[str] = []
    affected_tool_calls: set[int] = set()
    failures: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    try:
        tools = _parse_tools(sample.get("tools"))
    except (JSONDecodeError, TypeError) as error:
        tools = []
        failures.append(
            {
                "operation": "parse_tools",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    expected_recovered_names = audit_context.missing_tools.get(sample_id)
    if expected_recovered_names is not None:
        if tools:
            failures.append(
                {
                    "operation": RECOVER_SYSTEM_TOOL_DEFINITION,
                    "error_type": "ValueError",
                    "error": "审计要求恢复 tools，但当前顶层 tools 非空",
                }
            )
        else:
            try:
                tools = repair_missing_tools(
                    sample,
                    expected_recovered_names,
                )
            except (JSONDecodeError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "operation": RECOVER_SYSTEM_TOOL_DEFINITION,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            else:
                _append_unique(issues, MISSING_TOOLS)
                _append_unique(
                    issues, MALFORMED_SYSTEM_TOOL_DEFINITION
                )
                _append_unique(
                    repairs, RECOVER_SYSTEM_TOOL_DEFINITION
                )
                counts[RECOVER_SYSTEM_TOOL_DEFINITION] += 1

    repaired_sample["tools"] = tools
    available_tool_names = extract_tool_names(tools)
    parsed_tool_calls: list[dict[str, Any]] = []
    tool_call_index = 0

    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []

    for message_index, message in enumerate(conversations):
        if (
            not isinstance(message, dict)
            or message.get("from") != "gpt"
            or not isinstance(message.get("value"), str)
        ):
            continue

        for raw_content in TOOL_CALL_PATTERN.findall(message["value"]):
            key = (sample_id, tool_call_index)
            value: Any = None

            try:
                if key in audit_context.boundary_calls:
                    value = repair_boundary_literal_escape(raw_content)
                    _append_unique(issues, BOUNDARY_LITERAL_ESCAPE)
                    _append_unique(
                        repairs, REMOVE_BOUNDARY_LITERAL_ESCAPE
                    )
                    counts[REMOVE_BOUNDARY_LITERAL_ESCAPE] += 1
                elif key in audit_context.python_literal_calls:
                    value = repair_python_literal(raw_content)
                    _append_unique(issues, PYTHON_LITERAL_SYNTAX)
                    _append_unique(repairs, PYTHON_LITERAL_TO_OBJECT)
                    counts[PYTHON_LITERAL_TO_OBJECT] += 1
                else:
                    value = json.loads(raw_content)

                if not isinstance(value, dict):
                    raise TypeError("tool_call 解析结果必须是 object")

                verified_name = audit_context.verified_nested_names.get(key)
                if verified_name is not None:
                    repair_nested_tool_name(
                        value,
                        verified_name,
                        available_tool_names,
                    )
                    _append_unique(issues, NESTED_TOOL_NAME)
                    _append_unique(repairs, MOVE_NESTED_TOOL_NAME)
                    counts[MOVE_NESTED_TOOL_NAME] += 1

            except (JSONDecodeError, SyntaxError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "operation": "repair_tool_call",
                        "tool_call_index": tool_call_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

            parsed_tool_calls.append(
                {
                    "sample_id": sample_id,
                    "message_index": message_index,
                    "tool_call_index": tool_call_index,
                    "raw_content": raw_content,
                    "parsed_value": value,
                }
            )
            if (
                counts[REMOVE_BOUNDARY_LITERAL_ESCAPE]
                or counts[PYTHON_LITERAL_TO_OBJECT]
                or counts[MOVE_NESTED_TOOL_NAME]
            ):
                if (
                    key in audit_context.boundary_calls
                    or key in audit_context.python_literal_calls
                    or key in audit_context.verified_nested_names
                ):
                    affected_tool_calls.add(tool_call_index)

            tool_call_index += 1

    repaired_sample["tool_calls"] = parsed_tool_calls

    enum_evidence = {
        tool_name: evidence
        for (evidence_sample_id, tool_name), evidence
        in audit_context.invalid_enums.items()
        if evidence_sample_id == sample_id
    }
    if enum_evidence:
        try:
            removed, enum_affected_calls = (
                repair_invalid_enum_placeholders(tools, enum_evidence)
            )
        except (TypeError, ValueError) as error:
            failures.append(
                {
                    "operation": REMOVE_INVALID_ENUM_PLACEHOLDER,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            if removed:
                _append_unique(issues, INVALID_TOOL_SCHEMA)
                _append_unique(
                    repairs, REMOVE_INVALID_ENUM_PLACEHOLDER
                )
                counts[REMOVE_INVALID_ENUM_PLACEHOLDER] += removed
                affected_tool_calls.update(enum_affected_calls)

    if not repairs and not failures:
        return repaired_sample, []

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "status": "repair_failed" if failures else "repaired",
        "issues": issues,
        "repairs": repairs,
        "affected_tool_calls": sorted(affected_tool_calls),
        "repair_counts": dict(counts),
    }
    if failures:
        record["failures"] = failures

    return repaired_sample, [record]


def repair_hermes(
    samples: Iterable[dict[str, Any]],
    audit_context: RepairAuditContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """修复 Hermes 样本流；完整 repaired objects 只在内存中返回。"""
    repaired_samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for sample in samples:
        repaired_sample, sample_records = repair_hermes_sample(
            sample,
            audit_context,
        )
        repaired_samples.append(repaired_sample)
        records.extend(sample_records)

    return repaired_samples, records


def build_repair_summary(
    samples_total: int,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """汇总独立样本数与操作次数；两者不做相加推导。"""
    record_list = list(records)
    repair_counts: Counter[str] = Counter()

    for record in record_list:
        repair_counts.update(record.get("repair_counts", {}))

    failed_records = [
        record
        for record in record_list
        if record.get("status") == "repair_failed"
    ]
    repaired_records = [
        record
        for record in record_list
        if record.get("repairs")
    ]

    return {
        "dataset": "hermes_func_calling_singleturn",
        "samples_total": samples_total,
        "repaired_samples": len(repaired_records),
        "repair_counts": {
            REMOVE_BOUNDARY_LITERAL_ESCAPE: repair_counts[
                REMOVE_BOUNDARY_LITERAL_ESCAPE
            ],
            PYTHON_LITERAL_TO_OBJECT: repair_counts[
                PYTHON_LITERAL_TO_OBJECT
            ],
            MOVE_NESTED_TOOL_NAME: repair_counts[
                MOVE_NESTED_TOOL_NAME
            ],
            RECOVER_SYSTEM_TOOL_DEFINITION: repair_counts[
                RECOVER_SYSTEM_TOOL_DEFINITION
            ],
            "removed_invalid_enum_sites": repair_counts[
                REMOVE_INVALID_ENUM_PLACEHOLDER
            ],
        },
        "repair_failed": len(failed_records),
        "manual_review_samples": [
            record["sample_id"] for record in failed_records
        ],
    }


def write_repair_outputs(
    output_dir: str | Path,
    records: Iterable[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """只写 repair 日志与汇总，不写完整 repaired dataset。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "hermes_repair_records.jsonl"
    summary_path = directory / "hermes_repair_summary.json"

    with records_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return records_path, summary_path

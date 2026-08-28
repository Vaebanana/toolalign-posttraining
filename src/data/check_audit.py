"""运行并验收Hermes前六层正式审计。

本脚本依次执行strict failure语法分类、全量调用基础结构检查、全量tool name
与候选tools一致性检查、仅针对missing_tools的工具定义调查，以及全量调用的
arguments/Schema一致性检查，以及坏enum工具定义的定向调查。各层只更新问题
记录和累积issues，不执行修复。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit import (
    ARGUMENT_NAME_MISMATCH,
    ARGUMENT_TYPE_MISMATCH,
    BOUNDARY_LITERAL_ESCAPE,
    ENUM_MISMATCH,
    INVALID_ARGUMENTS_TYPE,
    INVALID_TOOL_CALL_TYPE,
    INVALID_TOOL_SCHEMA,
    MALFORMED_SYSTEM_TOOL_DEFINITION,
    MISSING_ARGUMENTS,
    MISSING_REQUIRED_ARGUMENT,
    MISSING_TOOL_NAME,
    MISSING_TOOLS,
    NESTED_TOOL_NAME,
    PYTHON_LITERAL_SYNTAX,
    UNCLASSIFIED_PARSE_FAILURE,
    UNKNOWN_TOOL,
    UNEXPECTED_ARGUMENT,
    audit_tool_calls_argument_schema_layer,
    audit_missing_tool_definitions_layer,
    audit_invalid_tool_schemas_layer,
    audit_tool_calls_tool_name_layer,
    audit_tool_calls_structure_layer,
    audit_tool_calls_syntax,
)
from parsers import load_hermes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hermes"
    / "func-calling-singleturn.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
LAYER_01_RECORDS_PATH = (
    OUTPUT_DIR / "hermes_audit_layer_01_json_syntax_records.jsonl"
)
LAYER_01_SUMMARY_PATH = (
    OUTPUT_DIR / "hermes_audit_layer_01_json_syntax_summary.json"
)
LAYER_02_RECORDS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_02_tool_call_structure_records.jsonl"
)
LAYER_02_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_02_tool_call_structure_summary.json"
)
LAYER_03_RECORDS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_03_tool_name_resolution_records.jsonl"
)
LAYER_03_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_03_tool_name_resolution_summary.json"
)
LAYER_04_RECORDS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_04_missing_tool_definition_records.jsonl"
)
LAYER_04_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_04_missing_tool_definition_summary.json"
)
LAYER_05_RECORDS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_05_argument_schema_consistency_records.jsonl"
)
LAYER_05_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_05_argument_schema_consistency_summary.json"
)
LAYER_06_RECORDS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_06_invalid_tool_schema_records.jsonl"
)
LAYER_06_DEFINITIONS_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_06_invalid_tool_schema_definitions.jsonl"
)
LAYER_06_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hermes_audit_layer_06_invalid_tool_schema_summary.json"
)
CUMULATIVE_RECORDS_PATH = (
    OUTPUT_DIR / "hermes_audit_cumulative_records.jsonl"
)
CUMULATIVE_SUMMARY_PATH = (
    OUTPUT_DIR / "hermes_audit_cumulative_summary.json"
)
LEGACY_OUTPUT_PATHS = (
    OUTPUT_DIR / "hermes_failed_tool_calls.jsonl",
    OUTPUT_DIR / "hermes_failed_tool_calls_summary.json",
    OUTPUT_DIR / "hermes_tool_call_audit.jsonl",
    OUTPUT_DIR / "hermes_tool_call_audit_summary.json",
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_cumulative_state(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """用固定文件名覆盖写入当前已完成层的累积审计状态。"""
    write_jsonl(CUMULATIVE_RECORDS_PATH, records)
    write_json(CUMULATIVE_SUMMARY_PATH, summary)


def main() -> None:
    print("正在执行Hermes严格解析……")
    samples = load_hermes(HERMES_PATH)
    tool_calls = [
        tool_call
        for sample in samples
        for tool_call in sample["tool_calls"]
    ]
    tools_by_sample_id: dict[Any, Any] = {}
    raw_samples_by_id: dict[Any, dict[str, Any]] = {}

    for sample in samples:
        sample_id = sample["id"]

        if sample_id in tools_by_sample_id:
            raise RuntimeError(f"Hermes sample id重复：{sample_id!r}")

        tools_by_sample_id[sample_id] = sample["tools_parse"]["value"]
        raw_samples_by_id[sample_id] = {
            "id": sample_id,
            "tools": sample["raw_tools"],
            "conversations": sample["raw_conversations"],
        }

    strict_success = sum(call["parse_success"] for call in tool_calls)
    strict_failed = len(tool_calls) - strict_success
    failed_tool_calls = [
        call
        for call in tool_calls
        if not call["parse_success"]
    ]
    parse_issue_names = {
        BOUNDARY_LITERAL_ESCAPE,
        PYTHON_LITERAL_SYNTAX,
        UNCLASSIFIED_PARSE_FAILURE,
    }
    structure_issue_names = {
        NESTED_TOOL_NAME,
        MISSING_TOOL_NAME,
        MISSING_ARGUMENTS,
        INVALID_ARGUMENTS_TYPE,
        INVALID_TOOL_CALL_TYPE,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("正在执行Audit Layer 01：JSON syntax……")
    layer_01_records = audit_tool_calls_syntax(failed_tool_calls)

    if len(layer_01_records) != strict_failed:
        raise RuntimeError(
            "Layer 01记录没有覆盖全部strict parse failures："
            f"failed={strict_failed}, "
            f"audited={len(layer_01_records)}"
        )

    parse_issue_counts: Counter[str] = Counter()

    for record in layer_01_records:
        if record["audit_layers_completed"] != [1]:
            raise RuntimeError(f"Layer 01完成状态错误：{record}")

        parse_issues = [
            issue
            for issue in record["issues"]
            if issue in parse_issue_names
        ]

        if record["parse"]["strict_json"]:
            if parse_issues:
                raise RuntimeError(f"严格解析成功记录出现语法问题：{record}")
        elif len(parse_issues) != 1:
            raise RuntimeError(
                "每条严格解析失败记录必须有一个语法主问题："
                f"{record}"
            )

        parse_issue_counts.update(parse_issues)

    classified_total = sum(parse_issue_counts.values())

    if classified_total != strict_failed:
        raise RuntimeError(
            "审计分类计数不一致："
            f"failed={strict_failed}, classified={classified_total}"
        )

    ordered_parse_counts = {
        BOUNDARY_LITERAL_ESCAPE: parse_issue_counts[
            BOUNDARY_LITERAL_ESCAPE
        ],
        PYTHON_LITERAL_SYNTAX: parse_issue_counts[PYTHON_LITERAL_SYNTAX],
        UNCLASSIFIED_PARSE_FAILURE: parse_issue_counts[
            UNCLASSIFIED_PARSE_FAILURE
        ],
    }
    layer_01_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 1,
        "audit_task": "json_syntax",
        "record_scope": "problem_tool_calls_found_through_layer_01",
        "audit_layers_completed": [1],
        "tool_call_tags": len(tool_calls),
        "audited_tool_calls": len(failed_tool_calls),
        "strict_parse_success": strict_success,
        "strict_parse_failed": strict_failed,
        "problem_record_count": strict_failed,
        "parse_failure_issue_counts": ordered_parse_counts,
        "audit_records": str(LAYER_01_RECORDS_PATH),
    }

    layer_01_problem_records = layer_01_records

    if len(layer_01_problem_records) != strict_failed:
        raise RuntimeError(
            "Layer 01问题记录数量与严格解析失败数不一致："
            f"problems={len(layer_01_problem_records)}, "
            f"failed={strict_failed}"
        )

    write_jsonl(LAYER_01_RECORDS_PATH, layer_01_problem_records)
    write_json(LAYER_01_SUMMARY_PATH, layer_01_summary)

    cumulative_layer_01_summary = dict(layer_01_summary)
    cumulative_layer_01_summary.update(
        {
            "output_kind": "cumulative_current_state",
            "current_audit_layer": 1,
            "current_audit_task": "json_syntax",
            "audit_records": str(CUMULATIVE_RECORDS_PATH),
            "layer_snapshot": str(LAYER_01_RECORDS_PATH),
        }
    )
    write_cumulative_state(
        layer_01_problem_records,
        cumulative_layer_01_summary,
    )

    print("正在执行Audit Layer 02：tool call structure……")
    layer_02_records = audit_tool_calls_structure_layer(
        tool_calls,
        layer_01_records,
    )

    structure_issue_counts: Counter[str] = Counter()
    layer_01_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_01_records
    }
    layer_02_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_02_records
    }

    if not layer_01_by_key.keys() <= layer_02_by_key.keys():
        raise RuntimeError("Layer 02丢失了Layer 01已有问题记录")

    for key, layer_02_record in layer_02_by_key.items():
        layer_01_record = layer_01_by_key.get(key)

        if layer_01_record is None:
            if layer_02_record["audit_layers_completed"] != [2]:
                raise RuntimeError(f"Layer 02新增记录状态错误：{layer_02_record}")
        else:
            if layer_02_record["audit_layers_completed"] != [1, 2]:
                raise RuntimeError(f"Layer 02更新记录状态错误：{layer_02_record}")

            inherited_issue_count = len(layer_01_record["issues"])

            if (
                layer_02_record["issues"][:inherited_issue_count]
                != layer_01_record["issues"]
            ):
                raise RuntimeError(
                    "Layer 02没有保留Layer 01 issues："
                    f"layer_01={layer_01_record}, "
                    f"layer_02={layer_02_record}"
                )

        structure_issue_counts.update(
            issue
            for issue in layer_02_record["issues"]
            if issue in structure_issue_names
        )

    ordered_structure_counts = {
        NESTED_TOOL_NAME: structure_issue_counts[NESTED_TOOL_NAME],
        MISSING_TOOL_NAME: structure_issue_counts[MISSING_TOOL_NAME],
        MISSING_ARGUMENTS: structure_issue_counts[MISSING_ARGUMENTS],
        INVALID_ARGUMENTS_TYPE: structure_issue_counts[
            INVALID_ARGUMENTS_TYPE
        ],
        INVALID_TOOL_CALL_TYPE: structure_issue_counts[
            INVALID_TOOL_CALL_TYPE
        ],
    }

    # Layer 02函数内部已经完成：更新已有问题记录，新增首次发现的问题。
    layer_02_problem_records = layer_02_records
    new_problem_count = len(layer_02_by_key.keys() - layer_01_by_key.keys())

    layer_02_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 2,
        "audit_task": "tool_call_structure",
        "record_scope": "problem_tool_calls_found_through_layer_02",
        "audit_layers_completed": [1, 2],
        "tool_call_tags": len(tool_calls),
        "audited_tool_calls": len(tool_calls),
        "inherited_problem_records": len(layer_01_records),
        "new_problem_records": new_problem_count,
        "problem_record_count": len(layer_02_problem_records),
        "inherited_parse_failure_issue_counts": ordered_parse_counts,
        "structure_issue_counts": ordered_structure_counts,
        "audit_records": str(LAYER_02_RECORDS_PATH),
    }

    write_jsonl(LAYER_02_RECORDS_PATH, layer_02_problem_records)
    write_json(LAYER_02_SUMMARY_PATH, layer_02_summary)

    cumulative_layer_02_summary = dict(layer_02_summary)
    cumulative_layer_02_summary.update(
        {
            "output_kind": "cumulative_current_state",
            "current_audit_layer": 2,
            "current_audit_task": "tool_call_structure",
            "audit_records": str(CUMULATIVE_RECORDS_PATH),
            "layer_snapshots": [
                str(LAYER_01_RECORDS_PATH),
                str(LAYER_02_RECORDS_PATH),
            ],
        }
    )
    write_cumulative_state(
        layer_02_problem_records,
        cumulative_layer_02_summary,
    )

    print("正在执行Audit Layer 03：tool name resolution……")
    layer_03_records, tool_name_results = (
        audit_tool_calls_tool_name_layer(
            tool_calls,
            tools_by_sample_id,
            layer_02_problem_records,
        )
    )

    if len(tool_name_results) != len(tool_calls):
        raise RuntimeError(
            "Layer 03没有审计全部tool calls："
            f"calls={len(tool_calls)}, "
            f"results={len(tool_name_results)}"
        )

    tool_name_counts: Counter[str] = Counter()
    nested_total = 0
    nested_verified = 0

    for result in tool_name_results:
        validation = result["validation"]

        if validation["verified"]:
            if result["issues"]:
                raise RuntimeError(f"工具名已验证但仍有名称问题：{result}")

            tool_name_counts["tool_name_verified"] += 1
        else:
            if len(result["issues"]) != 1:
                raise RuntimeError(
                    "工具名验证失败必须有一个明确问题："
                    f"{result}"
                )

            tool_name_counts[result["issues"][0]] += 1

        if validation["source"] == "nested_arguments_name":
            nested_total += 1

            if validation["verified"]:
                nested_verified += 1

    ordered_tool_name_counts = {
        "tool_name_verified": tool_name_counts["tool_name_verified"],
        UNKNOWN_TOOL: tool_name_counts[UNKNOWN_TOOL],
        MISSING_TOOLS: tool_name_counts[MISSING_TOOLS],
        MISSING_TOOL_NAME: tool_name_counts[MISSING_TOOL_NAME],
    }

    if sum(ordered_tool_name_counts.values()) != len(tool_calls):
        raise RuntimeError("Layer 03工具名验证计数不一致")

    nested_validation_counts = {
        "total": nested_total,
        "verified": nested_verified,
        "unverified": nested_total - nested_verified,
    }
    layer_02_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_02_problem_records
    }
    layer_03_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_03_records
    }

    if not layer_02_by_key.keys() <= layer_03_by_key.keys():
        raise RuntimeError("Layer 03丢失了Layer 02已有问题记录")

    for key, layer_03_record in layer_03_by_key.items():
        if "tool_name_validation" not in layer_03_record:
            raise RuntimeError(f"Layer 03记录缺少验证结果：{layer_03_record}")

        layer_02_record = layer_02_by_key.get(key)

        if layer_02_record is None:
            if layer_03_record["audit_layers_completed"] != [3]:
                raise RuntimeError(f"Layer 03新增记录状态错误：{layer_03_record}")
            continue

        if layer_03_record["audit_layers_completed"][-1] != 3:
            raise RuntimeError(f"Layer 03更新记录状态错误：{layer_03_record}")

        inherited_issue_count = len(layer_02_record["issues"])

        if (
            layer_03_record["issues"][:inherited_issue_count]
            != layer_02_record["issues"]
        ):
            raise RuntimeError(
                "Layer 03没有保留Layer 02 issues："
                f"layer_02={layer_02_record}, "
                f"layer_03={layer_03_record}"
            )

    new_layer_03_problems = len(
        layer_03_by_key.keys() - layer_02_by_key.keys()
    )
    layer_03_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 3,
        "audit_task": "tool_name_resolution",
        "record_scope": "problem_tool_calls_found_through_layer_03",
        "audit_layers_completed": [1, 2, 3],
        "tool_call_tags": len(tool_calls),
        "audited_tool_calls": len(tool_calls),
        "inherited_problem_records": len(layer_02_problem_records),
        "new_problem_records": new_layer_03_problems,
        "problem_record_count": len(layer_03_records),
        "tool_name_validation_counts": ordered_tool_name_counts,
        "nested_tool_name_validation": nested_validation_counts,
        "audit_records": str(LAYER_03_RECORDS_PATH),
    }

    write_jsonl(LAYER_03_RECORDS_PATH, layer_03_records)
    write_json(LAYER_03_SUMMARY_PATH, layer_03_summary)

    cumulative_layer_03_summary = dict(layer_03_summary)
    cumulative_layer_03_summary.update(
        {
            "output_kind": "cumulative_current_state",
            "current_audit_layer": 3,
            "current_audit_task": "tool_name_resolution",
            "audit_records": str(CUMULATIVE_RECORDS_PATH),
            "layer_snapshots": [
                str(LAYER_01_RECORDS_PATH),
                str(LAYER_02_RECORDS_PATH),
                str(LAYER_03_RECORDS_PATH),
            ],
        }
    )
    write_cumulative_state(
        layer_03_records,
        cumulative_layer_03_summary,
    )

    print("正在执行Audit Layer 04：missing tool definition investigation……")
    missing_tools_keys = {
        (record["sample_id"], record["tool_call_index"])
        for record in layer_03_records
        if MISSING_TOOLS in record["issues"]
    }
    missing_tools_calls = [
        tool_call
        for tool_call in tool_calls
        if (
            tool_call["sample_id"],
            tool_call["tool_call_index"],
        )
        in missing_tools_keys
    ]
    layer_04_records, sample_investigations = (
        audit_missing_tool_definitions_layer(
            missing_tools_calls,
            raw_samples_by_id,
            layer_03_records,
        )
    )

    if len(layer_04_records) != len(missing_tools_calls):
        raise RuntimeError("Layer 04没有覆盖全部missing_tools调用")

    affected_sample_ids = {
        tool_call["sample_id"]
        for tool_call in missing_tools_calls
    }

    if len(sample_investigations) != len(affected_sample_ids):
        raise RuntimeError("Layer 04 sample级调查数量不一致")

    top_level_state_counts = Counter(
        investigation["top_level_tools"]["state"]
        for investigation in sample_investigations
    )
    system_tools_present = sum(
        investigation["system_tools"]["present"]
        for investigation in sample_investigations
    )
    system_tools_parseable = sum(
        investigation["system_tools"]["strict_parseable"]
        for investigation in sample_investigations
    )
    system_tools_malformed = sum(
        investigation["system_tools"]["malformed"]
        for investigation in sample_investigations
    )
    recoverable_tool_definitions = sum(
        investigation["recovery"]["candidate_available"]
        for investigation in sample_investigations
    )
    malformed_issue_count = sum(
        MALFORMED_SYSTEM_TOOL_DEFINITION
        in investigation["issues"]
        for investigation in sample_investigations
    )
    layer_04_counts = {
        "missing_tool_calls": len(missing_tools_calls),
        "affected_samples": len(affected_sample_ids),
        "top_level_tools_missing": top_level_state_counts["missing"],
        "top_level_tools_empty": top_level_state_counts["empty"],
        "top_level_tools_parse_failed": top_level_state_counts[
            "parse_failed"
        ],
        "system_tools_present": system_tools_present,
        "system_tools_absent": (
            len(sample_investigations) - system_tools_present
        ),
        "system_tools_parseable": system_tools_parseable,
        "system_tools_malformed": system_tools_malformed,
        "recoverable_tool_definitions": recoverable_tool_definitions,
        "unrecoverable_tool_definitions": (
            len(sample_investigations) - recoverable_tool_definitions
        ),
        "malformed_system_tool_definition_issues": (
            malformed_issue_count
        ),
    }
    layer_03_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_03_records
    }
    layer_04_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_04_records
    }

    if layer_04_by_key.keys() != missing_tools_keys:
        raise RuntimeError("Layer 04输出范围不是全部且仅missing_tools调用")

    for key, layer_04_record in layer_04_by_key.items():
        layer_03_record = layer_03_by_key[key]
        inherited_issue_count = len(layer_03_record["issues"])

        if (
            layer_04_record["issues"][:inherited_issue_count]
            != layer_03_record["issues"]
        ):
            raise RuntimeError("Layer 04没有保留此前累积issues")

        if layer_04_record["audit_layers_completed"] != [1, 2, 3, 4]:
            raise RuntimeError("Layer 04记录完成状态错误")

        if "tool_definition_investigation" not in layer_04_record:
            raise RuntimeError("Layer 04记录缺少工具定义调查结果")

    cumulative_layer_04_records = [
        layer_04_by_key.get(
            (record["sample_id"], record["tool_call_index"]),
            record,
        )
        for record in layer_03_records
    ]
    layer_04_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 4,
        "audit_task": "missing_tool_definition_investigation",
        "record_scope": "targeted_missing_tools_calls",
        "audit_layers_completed": [1, 2, 3, 4],
        "audited_tool_calls": len(missing_tools_calls),
        "affected_samples": len(affected_sample_ids),
        "layer_record_count": len(layer_04_records),
        "cumulative_problem_record_count": len(
            cumulative_layer_04_records
        ),
        "investigation_counts": layer_04_counts,
        "audit_records": str(LAYER_04_RECORDS_PATH),
    }

    write_jsonl(LAYER_04_RECORDS_PATH, layer_04_records)
    write_json(LAYER_04_SUMMARY_PATH, layer_04_summary)

    cumulative_layer_04_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "output_kind": "cumulative_current_state",
        "current_audit_layer": 4,
        "current_audit_task": "missing_tool_definition_investigation",
        "audit_layers_completed": [1, 2, 3, 4],
        "problem_record_count": len(cumulative_layer_04_records),
        "audit_records": str(CUMULATIVE_RECORDS_PATH),
        "layer_results": {
            "layer_01_json_syntax": ordered_parse_counts,
            "layer_02_tool_call_structure": ordered_structure_counts,
            "layer_03_tool_name_resolution": {
                "tool_name_validation_counts": ordered_tool_name_counts,
                "nested_tool_name_validation": nested_validation_counts,
            },
            "layer_04_missing_tool_definition_investigation": (
                layer_04_counts
            ),
        },
        "layer_snapshots": [
            str(LAYER_01_RECORDS_PATH),
            str(LAYER_02_RECORDS_PATH),
            str(LAYER_03_RECORDS_PATH),
            str(LAYER_04_RECORDS_PATH),
        ],
    }
    write_cumulative_state(
        cumulative_layer_04_records,
        cumulative_layer_04_summary,
    )

    print("正在执行Audit Layer 05：argument/schema consistency……")
    diagnostic_tools_by_sample_id: dict[Any, Any] = {}

    for investigation in sample_investigations:
        diagnostic_candidates = investigation[
            "_diagnostic_candidate_values"
        ]

        if investigation["recovery"]["candidate_available"]:
            if len(diagnostic_candidates) != 1:
                raise RuntimeError(
                    "Layer 04唯一可恢复定义与诊断候选数量不一致"
                )

            diagnostic_tools_by_sample_id[
                investigation["sample_id"]
            ] = diagnostic_candidates[0]

    layer_05_records, argument_schema_results = (
        audit_tool_calls_argument_schema_layer(
            tool_calls,
            tools_by_sample_id,
            diagnostic_tools_by_sample_id,
            cumulative_layer_04_records,
        )
    )

    if len(argument_schema_results) != len(tool_calls):
        raise RuntimeError("Layer 05没有审计全部tool calls")

    schema_issue_names = {
        INVALID_TOOL_SCHEMA,
        ARGUMENT_NAME_MISMATCH,
        MISSING_REQUIRED_ARGUMENT,
        UNEXPECTED_ARGUMENT,
        ARGUMENT_TYPE_MISMATCH,
        ENUM_MISMATCH,
    }
    schema_issue_call_counts: Counter[str] = Counter()
    schema_issue_occurrence_counts: Counter[str] = Counter()
    definition_source_counts: Counter[str] = Counter()
    not_checked_reason_counts: Counter[str] = Counter()
    invalid_schema_reason_counts: Counter[str] = Counter()
    invalid_schema_sample_ids: set[Any] = set()
    partial_argument_checks = 0
    schema_checked = 0
    valid_against_schema = 0

    for tool_call, result in zip(tool_calls, argument_schema_results):
        source = result["tool_definition_source"]

        if source is not None:
            definition_source_counts[source] += 1

        if result["schema_checked"]:
            schema_checked += 1

        if result["valid_against_schema"]:
            valid_against_schema += 1

        if result.get("argument_validation_mode") == (
            "partial_excluding_invalid_enum_constraints"
        ):
            partial_argument_checks += 1

        result_schema_issues = [
            issue
            for issue in result["issues"]
            if issue in schema_issue_names
        ]
        schema_issue_call_counts.update(set(result_schema_issues))
        schema_issue_occurrence_counts.update(
            detail["issue"]
            for detail in result["issue_details"]
            if detail.get("issue") in schema_issue_names
        )

        if not result["schema_checked"]:
            reason = result.get("not_checked_reason")

            if reason is not None:
                not_checked_reason_counts[reason] += 1
            elif INVALID_TOOL_SCHEMA not in result["issues"]:
                raise RuntimeError(
                    "Layer 05未检查Schema但没有记录原因或invalid_tool_schema"
                )

        if INVALID_TOOL_SCHEMA in result["issues"]:
            invalid_schema_sample_ids.add(tool_call["sample_id"])

            for detail in result["issue_details"]:
                if detail.get("issue") != INVALID_TOOL_SCHEMA:
                    continue

                schema_errors = detail.get("schema_errors")

                if isinstance(schema_errors, list):
                    invalid_schema_reason_counts.update(
                        error.get("reason", "unclassified_schema_error")
                        for error in schema_errors
                    )
                else:
                    invalid_schema_reason_counts.update(
                        [detail.get("reason", "unclassified_schema_error")]
                    )

    ordered_schema_issue_call_counts = {
        INVALID_TOOL_SCHEMA: schema_issue_call_counts[INVALID_TOOL_SCHEMA],
        ARGUMENT_NAME_MISMATCH: schema_issue_call_counts[
            ARGUMENT_NAME_MISMATCH
        ],
        MISSING_REQUIRED_ARGUMENT: schema_issue_call_counts[
            MISSING_REQUIRED_ARGUMENT
        ],
        UNEXPECTED_ARGUMENT: schema_issue_call_counts[UNEXPECTED_ARGUMENT],
        ARGUMENT_TYPE_MISMATCH: schema_issue_call_counts[
            ARGUMENT_TYPE_MISMATCH
        ],
        ENUM_MISMATCH: schema_issue_call_counts[ENUM_MISMATCH],
    }
    ordered_schema_issue_occurrence_counts = {
        issue: schema_issue_occurrence_counts[issue]
        for issue in ordered_schema_issue_call_counts
    }
    layer_04_cumulative_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in cumulative_layer_04_records
    }
    layer_05_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_05_records
    }

    if not layer_04_cumulative_by_key.keys() <= layer_05_by_key.keys():
        raise RuntimeError("Layer 05丢失了此前累计问题记录")

    for key, layer_05_record in layer_05_by_key.items():
        if "argument_schema_validation" not in layer_05_record:
            raise RuntimeError("Layer 05问题记录缺少Schema审计结果")

        previous_record = layer_04_cumulative_by_key.get(key)

        if previous_record is None:
            if layer_05_record["audit_layers_completed"] != [5]:
                raise RuntimeError("Layer 05新增问题记录完成状态错误")
            continue

        inherited_issue_count = len(previous_record["issues"])

        if (
            layer_05_record["issues"][:inherited_issue_count]
            != previous_record["issues"]
        ):
            raise RuntimeError("Layer 05没有保留此前累积issues")

        expected_layers = [
            *previous_record["audit_layers_completed"],
            5,
        ]

        if layer_05_record["audit_layers_completed"] != expected_layers:
            raise RuntimeError("Layer 05累计记录完成状态错误")

    new_layer_05_problems = len(
        layer_05_by_key.keys() - layer_04_cumulative_by_key.keys()
    )
    layer_05_counts = {
        "total_tool_calls": len(tool_calls),
        "schema_checked": schema_checked,
        "schema_not_checked": len(tool_calls) - schema_checked,
        **ordered_schema_issue_call_counts,
        "valid_against_schema": valid_against_schema,
    }
    layer_05_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 5,
        "audit_task": "argument_schema_consistency",
        "record_scope": "problem_tool_calls_found_through_layer_05",
        "audited_tool_calls": len(tool_calls),
        "inherited_problem_records": len(cumulative_layer_04_records),
        "new_problem_records": new_layer_05_problems,
        "problem_record_count": len(layer_05_records),
        "validation_counts": layer_05_counts,
        "issue_occurrence_counts": (
            ordered_schema_issue_occurrence_counts
        ),
        "tool_definition_source_counts": dict(
            sorted(definition_source_counts.items())
        ),
        "not_checked_reason_counts": dict(
            sorted(not_checked_reason_counts.items())
        ),
        "invalid_tool_schema_diagnostics": {
            "affected_tool_calls": schema_issue_call_counts[
                INVALID_TOOL_SCHEMA
            ],
            "affected_samples": len(invalid_schema_sample_ids),
            "schema_error_occurrences": sum(
                invalid_schema_reason_counts.values()
            ),
            "schema_error_reason_counts": dict(
                sorted(invalid_schema_reason_counts.items())
            ),
            "partial_argument_checks": partial_argument_checks,
        },
        "audit_records": str(LAYER_05_RECORDS_PATH),
    }

    write_jsonl(LAYER_05_RECORDS_PATH, layer_05_records)
    write_json(LAYER_05_SUMMARY_PATH, layer_05_summary)

    cumulative_layer_05_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "output_kind": "cumulative_current_state",
        "current_audit_layer": 5,
        "current_audit_task": "argument_schema_consistency",
        "audit_layers_completed": [1, 2, 3, 4, 5],
        "problem_record_count": len(layer_05_records),
        "audit_records": str(CUMULATIVE_RECORDS_PATH),
        "layer_results": {
            "layer_01_json_syntax": ordered_parse_counts,
            "layer_02_tool_call_structure": ordered_structure_counts,
            "layer_03_tool_name_resolution": {
                "tool_name_validation_counts": ordered_tool_name_counts,
                "nested_tool_name_validation": nested_validation_counts,
            },
            "layer_04_missing_tool_definition_investigation": (
                layer_04_counts
            ),
            "layer_05_argument_schema_consistency": layer_05_counts,
        },
        "layer_snapshots": [
            str(LAYER_01_RECORDS_PATH),
            str(LAYER_02_RECORDS_PATH),
            str(LAYER_03_RECORDS_PATH),
            str(LAYER_04_RECORDS_PATH),
            str(LAYER_05_RECORDS_PATH),
        ],
    }
    write_cumulative_state(
        layer_05_records,
        cumulative_layer_05_summary,
    )

    print("正在执行Audit Layer 06：invalid tool schema investigation……")
    invalid_schema_keys = {
        (record["sample_id"], record["tool_call_index"])
        for record in layer_05_records
        if INVALID_TOOL_SCHEMA in record["issues"]
    }
    invalid_schema_calls = [
        tool_call
        for tool_call in tool_calls
        if (
            tool_call["sample_id"],
            tool_call["tool_call_index"],
        )
        in invalid_schema_keys
    ]
    layer_06_records, layer_06_definition_records = (
        audit_invalid_tool_schemas_layer(
            invalid_schema_calls,
            tools_by_sample_id,
            diagnostic_tools_by_sample_id,
            layer_05_records,
        )
    )

    if len(layer_06_records) != len(invalid_schema_calls):
        raise RuntimeError("Layer 06没有覆盖全部invalid_tool_schema调用")

    layer_05_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_05_records
    }
    layer_06_by_key = {
        (record["sample_id"], record["tool_call_index"]): record
        for record in layer_06_records
    }

    if layer_06_by_key.keys() != invalid_schema_keys:
        raise RuntimeError(
            "Layer 06输出范围不是全部且仅invalid_tool_schema调用"
        )

    for key, layer_06_record in layer_06_by_key.items():
        layer_05_record = layer_05_by_key[key]

        if layer_06_record["issues"] != layer_05_record["issues"]:
            raise RuntimeError("Layer 06不应改写Layer 05 issues")

        if layer_06_record["audit_layers_completed"] != [
            *layer_05_record["audit_layers_completed"],
            6,
        ]:
            raise RuntimeError("Layer 06记录完成状态错误")

        if "invalid_tool_schema_investigation" not in layer_06_record:
            raise RuntimeError("Layer 06记录缺少定向调查结果")

    placeholder_class_counts: Counter[str] = Counter()
    observation_evidence_counts: Counter[str] = Counter()
    call_reference_placeholder_counts: Counter[str] = Counter()
    unique_invalid_enum_sites = 0
    call_level_schema_error_references = 0
    remove_enum_candidate_sites = 0
    unresolved_enum_sites = 0
    definitions_with_only_remove_candidates = 0
    calls_with_only_remove_candidates = 0
    samples_with_only_remove_candidates: set[Any] = set()
    samples_with_unresolved_sites: set[Any] = set()

    for definition_record in layer_06_definition_records:
        sites = definition_record["invalid_enum_sites"]
        affected_call_count = len(
            definition_record["affected_tool_call_indexes"]
        )
        definition_all_candidates = definition_record[
            "all_sites_match_placeholder_pattern"
        ]
        unique_invalid_enum_sites += len(sites)
        call_level_schema_error_references += (
            len(sites) * affected_call_count
        )

        if definition_all_candidates:
            definitions_with_only_remove_candidates += 1
            calls_with_only_remove_candidates += affected_call_count
            samples_with_only_remove_candidates.add(
                definition_record["sample_id"]
            )
        else:
            samples_with_unresolved_sites.add(
                definition_record["sample_id"]
            )

        for site in sites:
            placeholder_class = site["placeholder_class"]
            placeholder_class_counts[placeholder_class] += 1
            call_reference_placeholder_counts[placeholder_class] += (
                affected_call_count
            )
            observation_evidence_counts[
                site["observation_evidence"]
            ] += 1

            if site["deterministic_repair_assessment"][
                "candidate_available"
            ]:
                remove_enum_candidate_sites += 1
            else:
                unresolved_enum_sites += 1

    # 同一sample若包含任一未决定义，就不能计入“全部定义均可候选删除”。
    samples_with_only_remove_candidates -= samples_with_unresolved_sites
    affected_layer_06_samples = {
        record["sample_id"] for record in layer_06_records
    }
    layer_06_counts = {
        "invalid_tool_schema_calls": len(invalid_schema_calls),
        "affected_samples": len(affected_layer_06_samples),
        "unique_tool_definitions": len(layer_06_definition_records),
        "call_level_schema_error_references": (
            call_level_schema_error_references
        ),
        "unique_invalid_enum_sites": unique_invalid_enum_sites,
        "singleton_null_enum_placeholder_sites": (
            placeholder_class_counts[
                "singleton_null_enum_placeholder"
            ]
        ),
        "empty_enum_placeholder_sites": placeholder_class_counts[
            "empty_enum_placeholder"
        ],
        "meaningful_or_mixed_invalid_enum_sites": (
            placeholder_class_counts[
                "meaningful_or_mixed_invalid_enum"
            ]
        ),
        "remove_enum_candidate_sites": remove_enum_candidate_sites,
        "unresolved_enum_sites": unresolved_enum_sites,
        "tool_definitions_with_only_remove_candidates": (
            definitions_with_only_remove_candidates
        ),
        "calls_with_only_remove_candidates": (
            calls_with_only_remove_candidates
        ),
        "samples_with_only_remove_candidates": len(
            samples_with_only_remove_candidates
        ),
    }

    if call_level_schema_error_references != sum(
        invalid_schema_reason_counts.values()
    ):
        raise RuntimeError(
            "Layer 06调用级Schema错误引用数与Layer 05不一致"
        )

    cumulative_layer_06_records = [
        layer_06_by_key.get(
            (record["sample_id"], record["tool_call_index"]),
            record,
        )
        for record in layer_05_records
    ]
    layer_06_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "audit_layer": 6,
        "audit_task": "invalid_tool_schema_investigation",
        "record_scope": "targeted_invalid_tool_schema_calls",
        "layer_record_count": len(layer_06_records),
        "cumulative_problem_record_count": len(
            cumulative_layer_06_records
        ),
        "investigation_counts": layer_06_counts,
        "observation_evidence_counts": dict(
            sorted(observation_evidence_counts.items())
        ),
        "call_reference_placeholder_counts": dict(
            sorted(call_reference_placeholder_counts.items())
        ),
        "repair_candidate_policy": {
            "candidate_operation": "remove_enum",
            "preserve_declared_type": True,
            "widen_type_to_include_null": False,
            "invent_enum_values": False,
            "applied_to_data": False,
        },
        "audit_records": str(LAYER_06_RECORDS_PATH),
        "definition_investigations": str(
            LAYER_06_DEFINITIONS_PATH
        ),
    }

    write_jsonl(LAYER_06_RECORDS_PATH, layer_06_records)
    write_jsonl(
        LAYER_06_DEFINITIONS_PATH,
        layer_06_definition_records,
    )
    write_json(LAYER_06_SUMMARY_PATH, layer_06_summary)

    cumulative_layer_06_summary = {
        "dataset": "hermes_func_calling_singleturn",
        "output_kind": "cumulative_current_state",
        "current_audit_layer": 6,
        "current_audit_task": "invalid_tool_schema_investigation",
        "audit_layers_completed": [1, 2, 3, 4, 5, 6],
        "problem_record_count": len(cumulative_layer_06_records),
        "audit_records": str(CUMULATIVE_RECORDS_PATH),
        "layer_results": {
            **cumulative_layer_05_summary["layer_results"],
            "layer_06_invalid_tool_schema_investigation": (
                layer_06_counts
            ),
        },
        "layer_snapshots": [
            *cumulative_layer_05_summary["layer_snapshots"],
            str(LAYER_06_RECORDS_PATH),
        ],
    }
    write_cumulative_state(
        cumulative_layer_06_records,
        cumulative_layer_06_summary,
    )

    for legacy_path in LEGACY_OUTPUT_PATHS:
        legacy_path.unlink(missing_ok=True)

    print("Hermes tool calls:", len(tool_calls))
    print("Hermes strict parse success:", strict_success)
    print("Hermes failed tool calls:", strict_failed)
    print("\nParse failure audit:")

    for issue, count in ordered_parse_counts.items():
        print(f"{issue}: {count}")

    print("\nHermes tool call structure audit:")
    print("total tool calls:", len(tool_calls))

    for issue, count in ordered_structure_counts.items():
        print(f"{issue}: {count}")

    print("\nAudit Layer 03: tool name resolution")
    print("total tool calls:", len(tool_calls))
    print("tool name verified:", ordered_tool_name_counts["tool_name_verified"])
    print("unknown_tool:", ordered_tool_name_counts[UNKNOWN_TOOL])
    print("missing_tools:", ordered_tool_name_counts[MISSING_TOOLS])
    print("missing_tool_name:", ordered_tool_name_counts[MISSING_TOOL_NAME])
    print("nested total:", nested_validation_counts["total"])
    print("nested verified:", nested_validation_counts["verified"])
    print("nested unverified:", nested_validation_counts["unverified"])

    print("\nAudit Layer 04: missing tool definitions")

    for name, count in layer_04_counts.items():
        print(f"{name}: {count}")

    print("\nAudit Layer 05: argument/schema consistency")

    for name, count in layer_05_counts.items():
        print(f"{name}: {count}")

    print(
        "invalid_tool_schema affected samples:",
        len(invalid_schema_sample_ids),
    )
    print(
        "invalid_tool_schema error occurrences:",
        sum(invalid_schema_reason_counts.values()),
    )

    print("\nAudit Layer 06: invalid tool schema investigation")

    for name, count in layer_06_counts.items():
        print(f"{name}: {count}")

    print(f"\nLayer 01 records：{LAYER_01_RECORDS_PATH}")
    print(f"Layer 01 summary：{LAYER_01_SUMMARY_PATH}")
    print(f"Layer 02 records：{LAYER_02_RECORDS_PATH}")
    print(f"Layer 02 summary：{LAYER_02_SUMMARY_PATH}")
    print(f"Layer 03 records：{LAYER_03_RECORDS_PATH}")
    print(f"Layer 03 summary：{LAYER_03_SUMMARY_PATH}")
    print(f"Layer 04 records：{LAYER_04_RECORDS_PATH}")
    print(f"Layer 04 summary：{LAYER_04_SUMMARY_PATH}")
    print(f"Layer 05 records：{LAYER_05_RECORDS_PATH}")
    print(f"Layer 05 summary：{LAYER_05_SUMMARY_PATH}")
    print(f"Layer 06 records：{LAYER_06_RECORDS_PATH}")
    print(f"Layer 06 definitions：{LAYER_06_DEFINITIONS_PATH}")
    print(f"Layer 06 summary：{LAYER_06_SUMMARY_PATH}")
    print(f"Cumulative records：{CUMULATIVE_RECORDS_PATH}")
    print(f"Cumulative summary：{CUMULATIVE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

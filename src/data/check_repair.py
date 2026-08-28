"""执行并验收 Hermes Day 01 第一阶段 deterministic repair。"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from hermes_utils import extract_tool_names, iter_invalid_enum_sites
from parsers import load_json_array
from repair import (
    MOVE_NESTED_TOOL_NAME,
    PYTHON_LITERAL_TO_OBJECT,
    RECOVER_SYSTEM_TOOL_DEFINITION,
    RepairAuditContext,
    REMOVE_BOUNDARY_LITERAL_ESCAPE,
    build_repair_summary,
    load_repair_audit_context,
    repair_hermes,
    write_repair_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hermes"
    / "func-calling-singleturn.json"
)
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "repair"

EXPECTED_SAMPLES = 1893
EXPECTED_TOOL_CALLS = 2981
EXPECTED_REPAIR_COUNTS = {
    REMOVE_BOUNDARY_LITERAL_ESCAPE: 6,
    PYTHON_LITERAL_TO_OBJECT: 787,
    MOVE_NESTED_TOOL_NAME: 793,
    RECOVER_SYSTEM_TOOL_DEFINITION: 61,
    "removed_invalid_enum_sites": 131,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _definition(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None

    function = tool.get("function")
    return function if isinstance(function, dict) else tool


def _validate_repaired_samples(
    samples: list[dict[str, Any]],
    context: RepairAuditContext,
) -> None:
    tool_call_count = 0

    for sample in samples:
        tools = sample.get("tools")
        tool_calls = sample.get("tool_calls")

        if not isinstance(tools, list):
            raise AssertionError(
                f"repaired tools 不是 list：{sample.get('id')!r}"
            )
        if not isinstance(tool_calls, list):
            raise AssertionError(
                f"repaired tool_calls 不是 list：{sample.get('id')!r}"
            )

        names = extract_tool_names(tools)
        tool_call_count += len(tool_calls)

        for call in tool_calls:
            value = call.get("parsed_value")
            if not isinstance(value, dict):
                raise AssertionError(
                    "存在未解析为 object 的 tool_call："
                    f"{sample.get('id')!r}/{call.get('tool_call_index')}"
                )

            name = value.get("name")
            arguments = value.get("arguments")
            if not isinstance(name, str) or name not in names:
                raise AssertionError(
                    "tool_call name 未被 repaired tools 验证："
                    f"{sample.get('id')!r}/{call.get('tool_call_index')}"
                )
            if not isinstance(arguments, dict):
                raise AssertionError(
                    "tool_call arguments 不是 object："
                    f"{sample.get('id')!r}/{call.get('tool_call_index')}"
                )

        audited_enum_paths = {
            tool_name: set(evidence.paths)
            for (evidence_sample_id, tool_name), evidence
            in context.invalid_enums.items()
            if evidence_sample_id == sample.get("id")
        }

        for tool in tools:
            definition = _definition(tool)
            if definition is None:
                continue

            tool_name = definition.get("name")
            expected_removed_paths = audited_enum_paths.get(tool_name)
            if not expected_removed_paths:
                continue

            remaining_paths = {
                site["schema_path"]
                for site in iter_invalid_enum_sites(
                    definition.get("parameters")
                )
            }
            not_removed = expected_removed_paths & remaining_paths
            if not_removed:
                raise AssertionError(
                    "仍存在已确认的 enum placeholder："
                    f"{sample.get('id')!r}/{tool_name!r}/"
                    f"{sorted(not_removed)}"
                )

    if tool_call_count != EXPECTED_TOOL_CALLS:
        raise AssertionError(
            f"tool_call 总数错误：{tool_call_count} != {EXPECTED_TOOL_CALLS}"
        )


def main() -> None:
    raw_hash_before = _sha256(HERMES_PATH)
    raw_samples = load_json_array(HERMES_PATH)
    raw_snapshot = copy.deepcopy(raw_samples)
    context = load_repair_audit_context(AUDIT_DIR)

    repaired_samples, records = repair_hermes(raw_samples, context)
    summary = build_repair_summary(len(raw_samples), records)
    records_path, summary_path = write_repair_outputs(
        OUTPUT_DIR,
        records,
        summary,
    )

    if len(raw_samples) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"样本数错误：{len(raw_samples)} != {EXPECTED_SAMPLES}"
        )
    if raw_samples != raw_snapshot:
        raise AssertionError("repair 修改了传入的 raw samples")
    if _sha256(HERMES_PATH) != raw_hash_before:
        raise AssertionError("repair 修改了磁盘上的 Hermes raw 文件")
    if summary["repair_counts"] != EXPECTED_REPAIR_COUNTS:
        raise AssertionError(
            "repair 操作数与审计不一致："
            f"{summary['repair_counts']}"
        )
    if summary["repair_failed"] != 0:
        raise AssertionError(
            f"存在 repair_failed：{summary['manual_review_samples']}"
        )

    _validate_repaired_samples(repaired_samples, context)

    print("Hermes deterministic repair 验收通过")
    print(f"samples_total: {summary['samples_total']}")
    print(f"repaired_samples: {summary['repaired_samples']}")
    print(f"repair_counts: {summary['repair_counts']}")
    print(f"repair_failed: {summary['repair_failed']}")
    print(f"records: {records_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

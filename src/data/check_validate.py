"""执行并验收 Day 01 Stage 3 Final Validation。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validate import (
    DROP,
    KEEP,
    MANUAL_REVIEW,
    REPAIRED_KEEP,
    summarize_validation,
    validate_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = (
    PROJECT_ROOT / "data" / "intermediate" / "normalized_candidates.jsonl"
)
REPAIR_RECORDS_PATH = (
    PROJECT_ROOT / "outputs" / "repair" / "hermes_repair_records.jsonl"
)
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"
VALIDATION_RECORDS_PATH = VALIDATION_DIR / "validation_records.jsonl"
VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "validation_summary.json"
PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "normalized_all.jsonl"
)
EXPECTED_CANDIDATES = 61893


def _load_repaired_sample_ids(path: Path) -> set[str]:
    repaired_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("repairs"):
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str):
                    raise TypeError(
                        f"{path}:{line_number} repair sample_id 非字符串"
                    )
                repaired_ids.add(f"hermes_{sample_id}")

    return repaired_ids


def _json_decode_drop_record(
    line_number: int,
    error: json.JSONDecodeError,
) -> dict[str, Any]:
    return {
        "sample_id": None,
        "source": "unknown",
        "status": DROP,
        "issues": ["invalid_canonical_json"],
        "validation_errors": [
            {
                "issue": "invalid_canonical_json",
                "message": str(error),
                "line_number": line_number,
            }
        ],
    }


def _validate_status_provenance(
    record: dict[str, Any],
    repaired_sample_ids: set[str],
) -> None:
    sample_id = record.get("sample_id")
    status = record["status"]

    if status == REPAIRED_KEEP and sample_id not in repaired_sample_ids:
        raise AssertionError(
            f"repaired_keep 缺少 repair 记录：{sample_id!r}"
        )
    if status == KEEP and sample_id in repaired_sample_ids:
        raise AssertionError(
            f"经过 Repair 的样本被错误标成 keep：{sample_id!r}"
        )


def _verify_processed_file(
    path: Path,
    expected_lines: int,
) -> None:
    line_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            sample = json.loads(line)
            line_count += 1
            forbidden = {
                "status",
                "issues",
                "repairs",
                "validation_errors",
                "processing",
            }
            leaked = forbidden & set(sample)
            if leaked:
                raise AssertionError(
                    f"processed:{line_number} 泄漏状态字段：{sorted(leaked)}"
                )

    if line_count != expected_lines:
        raise AssertionError(
            f"processed 行数错误：{line_count} != {expected_lines}"
        )


def main() -> None:
    repaired_sample_ids = _load_repaired_sample_ids(REPAIR_RECORDS_PATH)
    records: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        CANDIDATES_PATH.open("r", encoding="utf-8") as candidates_file,
        VALIDATION_RECORDS_PATH.open("w", encoding="utf-8") as records_file,
        PROCESSED_PATH.open("w", encoding="utf-8") as processed_file,
    ):
        for line_number, line in enumerate(candidates_file, start=1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                record = _json_decode_drop_record(line_number, error)
                usable = False
            else:
                sample_id = (
                    sample.get("sample_id")
                    if isinstance(sample, dict)
                    else None
                )
                if isinstance(sample_id, str) and sample_id in seen_sample_ids:
                    record = {
                        "sample_id": sample_id,
                        "source": sample.get("source", "unknown"),
                        "status": DROP,
                        "issues": ["duplicate_sample_id"],
                        "validation_errors": [
                            {
                                "issue": "duplicate_sample_id",
                                "message": "canonical sample_id is duplicated",
                                "line_number": line_number,
                            }
                        ],
                    }
                    usable = False
                else:
                    record, usable = validate_sample(
                        sample,
                        repaired_sample_ids,
                    )

                if isinstance(sample_id, str):
                    seen_sample_ids.add(sample_id)

            _validate_status_provenance(record, repaired_sample_ids)
            records.append(record)
            records_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            if usable:
                # 写回原始 canonical JSON 行，不注入 status/issues/repair 信息。
                processed_file.write(line)

    summary = summarize_validation(records)
    with VALIDATION_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if summary["input_candidates"] != EXPECTED_CANDIDATES:
        raise AssertionError(
            "validation candidate 数错误："
            f"{summary['input_candidates']} != {EXPECTED_CANDIDATES}"
        )
    if sum(summary["status_counts"].values()) != EXPECTED_CANDIDATES:
        raise AssertionError("validation 状态数之和与输入不一致")
    if len(repaired_sample_ids) != 837:
        raise AssertionError(
            f"repair 样本关联数错误：{len(repaired_sample_ids)} != 837"
        )

    _verify_processed_file(PROCESSED_PATH, summary["final_usable"])

    print("Final Validation 验收通过")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"records: {VALIDATION_RECORDS_PATH}")
    print(f"summary: {VALIDATION_SUMMARY_PATH}")
    print(f"processed: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()

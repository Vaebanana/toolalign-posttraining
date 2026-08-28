"""执行并验收 Day 01 Stage 4 Split。"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from split import (
    DEV_SEEN,
    MIN_UNSEEN_TOOL_SAMPLES,
    SEED,
    SEEN_TRAIN_RATIO,
    SPLIT_NAMES,
    TEST_SEEN,
    TEST_UNSEEN_TOOLS,
    TRAIN,
    UNSEEN_TARGET_RATIO,
    build_group_key,
    collect_tool_names,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "normalized_all.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "split"
SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"
UNSEEN_TOOLS_PATH = OUTPUT_DIR / "unseen_tools.json"
SPLIT_PATHS = {
    TRAIN: PROCESSED_DIR / "train.jsonl",
    DEV_SEEN: PROCESSED_DIR / "dev_seen.jsonl",
    TEST_SEEN: PROCESSED_DIR / "test_seen.jsonl",
    TEST_UNSEEN_TOOLS: PROCESSED_DIR / "test_unseen_tools.jsonl",
}
EXPECTED_INPUT_SAMPLES = 59334


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 必须是 JSON object")
            samples.append(value)

    return samples


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _sample_ids(samples: Iterable[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str):
            raise AssertionError("split sample_id 必须是字符串")
        if sample_id in ids:
            raise AssertionError(f"split 内 sample_id 重复：{sample_id}")
        ids.add(sample_id)
    return ids


def _split_signature(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    return {
        name: [sample["sample_id"] for sample in splits[name]]
        for name in SPLIT_NAMES
    }


def _validate_splits(
    input_samples: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    heldout_tools: set[str],
    tool_frequency: Counter[str],
) -> None:
    input_ids = _sample_ids(input_samples)
    ids_by_split = {
        name: _sample_ids(samples) for name, samples in splits.items()
    }

    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = ids_by_split[left] & ids_by_split[right]
        if overlap:
            raise AssertionError(
                f"sample_id 跨 split：{left}/{right}: {len(overlap)}"
            )

    output_ids = set().union(*ids_by_split.values())
    if output_ids != input_ids:
        raise AssertionError(
            "split 未做到全量且唯一覆盖："
            f"missing={len(input_ids-output_ids)}, "
            f"extra={len(output_ids-input_ids)}"
        )

    groups_by_split = {
        name: {build_group_key(sample) for sample in samples}
        for name, samples in splits.items()
    }
    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = groups_by_split[left] & groups_by_split[right]
        if overlap:
            raise AssertionError(
                f"query group 跨 split：{left}/{right}: {len(overlap)}"
            )

    train_tools: set[str] = set()
    for sample in splits[TRAIN]:
        train_tools.update(collect_tool_names(sample))
    leaked_heldout = heldout_tools & train_tools
    if leaked_heldout:
        raise AssertionError(
            f"held-out tools 泄漏到 train：{sorted(leaked_heldout)[:10]}"
        )

    for split_name in (TRAIN, DEV_SEEN, TEST_SEEN):
        for sample in splits[split_name]:
            if collect_tool_names(sample) & heldout_tools:
                raise AssertionError(
                    f"held-out tool 样本落入 {split_name}："
                    f"{sample['sample_id']}"
                )

    if any(
        tool_frequency[name] < MIN_UNSEEN_TOOL_SAMPLES
        for name in heldout_tools
    ):
        raise AssertionError("held-out tools 包含低于最低频次的工具")

    unseen_size = len(splits[TEST_UNSEEN_TOOLS])
    if not 2500 <= unseen_size <= 3500:
        raise AssertionError(
            f"test_unseen_tools 数量偏离约定范围：{unseen_size}"
        )


def _source_distribution(
    samples: Iterable[dict[str, Any]],
) -> dict[str, int]:
    counts = Counter(sample.get("source") for sample in samples)
    return {
        "total": sum(counts.values()),
        "xlam": counts["xlam"],
        "hermes": counts["hermes"],
    }


def _verify_written_files(
    paths: dict[str, Path],
    expected_counts: dict[str, int],
) -> None:
    for split_name, path in paths.items():
        count = 0
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                sample = json.loads(line)
                count += 1
                leaked = {
                    "status",
                    "issues",
                    "repairs",
                    "validation_errors",
                    "processing",
                    "group_key",
                } & set(sample)
                if leaked:
                    raise AssertionError(
                        f"{path}:{line_number} 泄漏辅助字段：{sorted(leaked)}"
                    )
        if count != expected_counts[split_name]:
            raise AssertionError(
                f"{split_name} 文件行数错误："
                f"{count} != {expected_counts[split_name]}"
            )


def main() -> None:
    print("正在读取 normalized_all.jsonl……")
    samples = _load_jsonl(INPUT_PATH)
    if len(samples) != EXPECTED_INPUT_SAMPLES:
        raise AssertionError(
            f"Split 输入数量错误：{len(samples)} != {EXPECTED_INPUT_SAMPLES}"
        )

    print("正在选择 held-out tools 并执行 group split……")
    result = split_dataset(samples, seed=SEED)
    _validate_splits(
        samples,
        result.splits,
        result.heldout_tools,
        result.tool_frequency,
    )

    # 相同输入与 seed 必须得到完全相同的工具集合和样本顺序。
    repeated = split_dataset(samples, seed=SEED)
    if repeated.heldout_tools != result.heldout_tools or (
        _split_signature(repeated.splits) != _split_signature(result.splits)
    ):
        raise AssertionError("SEED=42 的 split 结果不可复现")

    split_counts = {
        name: len(result.splits[name]) for name in SPLIT_NAMES
    }
    if sum(split_counts.values()) != len(samples):
        raise AssertionError("四份 split 数量之和不等于输入")

    print("正在写入四份 split……")
    for split_name, path in SPLIT_PATHS.items():
        _write_jsonl(path, result.splits[split_name])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_samples": len(samples),
        "seed": SEED,
        "config": {
            "group_key": "sha256(normalize_text(user_content))",
            "min_unseen_tool_samples": MIN_UNSEEN_TOOL_SAMPLES,
            "unseen_target_ratio": UNSEEN_TARGET_RATIO,
            "seen_train_ratio": SEEN_TRAIN_RATIO,
            "seen_dev_ratio": 0.05,
            "seen_test_ratio": 0.05,
        },
        "heldout_tool_count": len(result.heldout_tools),
        "target_unseen_samples": result.target_unseen_samples,
        "splits": {
            name: _source_distribution(result.splits[name])
            for name in SPLIT_NAMES
        },
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    unseen_tools = {
        "seed": SEED,
        "min_tool_samples": MIN_UNSEEN_TOOL_SAMPLES,
        "target_samples": result.target_unseen_samples,
        "actual_samples": len(result.splits[TEST_UNSEEN_TOOLS]),
        "tools": sorted(result.heldout_tools),
    }
    with UNSEEN_TOOLS_PATH.open("w", encoding="utf-8") as file:
        json.dump(unseen_tools, file, ensure_ascii=False, indent=2)
        file.write("\n")

    _verify_written_files(SPLIT_PATHS, split_counts)

    print("Split 验收通过")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"summary: {SUMMARY_PATH}")
    print(f"unseen tools: {UNSEEN_TOOLS_PATH}")


if __name__ == "__main__":
    main()

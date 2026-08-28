"""执行并验收 Day 01 第二阶段 Normalize。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from normalize import (
    UNIFIED_SYSTEM_PROMPT,
    normalize_dataset,
    normalize_hermes_sample,
    normalize_xlam_sample,
)
from parsers import load_json_array, load_xlam
from repair import load_repair_audit_context, repair_hermes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XLAM_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xlam"
    / "xlam_function_calling_60k.json"
)
HERMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hermes"
    / "func-calling-singleturn.json"
)
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "normalize"
CANDIDATES_PATH = INTERMEDIATE_DIR / "normalized_candidates.jsonl"
ERRORS_PATH = OUTPUT_DIR / "normalize_errors.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "normalize_summary.json"

CANONICAL_FIELDS = {
    "sample_id",
    "source",
    "messages",
    "tools",
    "assistant",
    "metadata",
}
SOURCE_PROTOCOL_FIELDS = {
    "query",
    "answers",
    "conversations",
    "from",
    "value",
}


def _validate_canonical_sample(sample: dict[str, Any]) -> None:
    sample_id = sample.get("sample_id")
    if set(sample) != CANONICAL_FIELDS:
        raise AssertionError(
            f"canonical 顶层字段错误：{sample_id!r}: {set(sample)}"
        )
    if not isinstance(sample_id, str) or not sample_id:
        raise AssertionError("sample_id 必须是非空字符串")
    if sample.get("source") not in {"xlam", "hermes"}:
        raise AssertionError(f"source 错误：{sample_id!r}")
    if SOURCE_PROTOCOL_FIELDS & set(sample):
        raise AssertionError(f"残留源协议字段：{sample_id!r}")

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise AssertionError(f"messages 结构错误：{sample_id!r}")
    if messages[0] != {
        "role": "system",
        "content": UNIFIED_SYSTEM_PROMPT,
    }:
        raise AssertionError(f"system prompt 未统一：{sample_id!r}")

    for message in messages:
        if set(message) != {"role", "content"}:
            raise AssertionError(f"message 字段错误：{sample_id!r}")
        if message["role"] not in {"system", "user"}:
            raise AssertionError(f"message role 错误：{sample_id!r}")
        if not isinstance(message["content"], str):
            raise AssertionError(f"message content 错误：{sample_id!r}")
        if "<tools>" in message["content"] or "<tool_call>" in message[
            "content"
        ]:
            raise AssertionError(f"message 残留 Hermes 协议：{sample_id!r}")

    tools = sample.get("tools")
    if not isinstance(tools, list):
        raise AssertionError(f"tools 不是 array：{sample_id!r}")
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {
            "type",
            "function",
        }:
            raise AssertionError(f"tool wrapper 错误：{sample_id!r}")
        if tool["type"] != "function":
            raise AssertionError(f"tool type 错误：{sample_id!r}")
        function = tool["function"]
        if not isinstance(function, dict) or set(function) != {
            "name",
            "description",
            "parameters",
        }:
            raise AssertionError(f"tool function 字段错误：{sample_id!r}")
        if not isinstance(function["name"], str):
            raise AssertionError(f"tool name 错误：{sample_id!r}")
        if not isinstance(function["description"], str):
            raise AssertionError(f"tool description 错误：{sample_id!r}")
        if not isinstance(function["parameters"], dict):
            raise AssertionError(f"tool parameters 错误：{sample_id!r}")

    assistant = sample.get("assistant")
    if not isinstance(assistant, dict) or set(assistant) != {
        "content",
        "tool_calls",
    }:
        raise AssertionError(f"assistant 结构错误：{sample_id!r}")
    content = assistant["content"]
    calls = assistant["tool_calls"]
    if content is not None and not isinstance(content, str):
        raise AssertionError(f"assistant content 错误：{sample_id!r}")
    if not isinstance(calls, list):
        raise AssertionError(f"assistant tool_calls 错误：{sample_id!r}")
    if calls and content is not None:
        raise AssertionError(f"assistant content/calls 冲突：{sample_id!r}")
    if isinstance(content, str) and (
        "<tools>" in content or "<tool_call>" in content
    ):
        raise AssertionError(f"assistant 残留 Hermes 协议：{sample_id!r}")

    for call in calls:
        if not isinstance(call, dict) or set(call) != {
            "type",
            "function",
        }:
            raise AssertionError(f"tool_call wrapper 错误：{sample_id!r}")
        if call["type"] != "function":
            raise AssertionError(f"tool_call type 错误：{sample_id!r}")
        function = call["function"]
        if not isinstance(function, dict) or set(function) != {
            "name",
            "arguments",
        }:
            raise AssertionError(f"tool_call function 错误：{sample_id!r}")
        if not isinstance(function["name"], str):
            raise AssertionError(f"tool_call name 错误：{sample_id!r}")
        if not isinstance(function["arguments"], dict):
            raise AssertionError(f"tool_call arguments 错误：{sample_id!r}")

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "category",
        "subcategory",
        "task",
    }:
        raise AssertionError(f"metadata 结构错误：{sample_id!r}")


def _validate_all(samples: Iterable[dict[str, Any]]) -> Counter[str]:
    source_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()

    for sample in samples:
        _validate_canonical_sample(sample)
        sample_id = sample["sample_id"]
        if sample_id in sample_ids:
            raise AssertionError(f"canonical sample_id 重复：{sample_id}")
        sample_ids.add(sample_id)
        source_counts[sample["source"]] += 1

    return source_counts


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    print("正在解析并 Normalize xLAM……")
    xlam_samples = load_xlam(XLAM_PATH)
    normalized_xlam, xlam_errors = normalize_dataset(
        xlam_samples,
        "xlam",
        normalize_xlam_sample,
    )

    print("正在 Repair 并 Normalize Hermes……")
    raw_hermes = load_json_array(HERMES_PATH)
    repair_context = load_repair_audit_context(AUDIT_DIR)
    repaired_hermes, _ = repair_hermes(raw_hermes, repair_context)
    normalized_hermes, hermes_errors = normalize_dataset(
        repaired_hermes,
        "hermes",
        normalize_hermes_sample,
    )

    candidates = [*normalized_xlam, *normalized_hermes]
    errors = [*xlam_errors, *hermes_errors]
    source_counts = _validate_all(candidates)

    summary = {
        "xlam": {
            "input_samples": len(xlam_samples),
            "normalized": len(normalized_xlam),
            "failed": len(xlam_errors),
        },
        "hermes": {
            "input_samples": len(raw_hermes),
            "normalized": len(normalized_hermes),
            "failed": len(hermes_errors),
        },
        "total_normalized": len(candidates),
        "total_failed": len(errors),
    }

    if source_counts != Counter(
        {"xlam": len(normalized_xlam), "hermes": len(normalized_hermes)}
    ):
        raise AssertionError(f"source 计数不一致：{source_counts}")

    _write_jsonl(CANDIDATES_PATH, candidates)
    _write_jsonl(ERRORS_PATH, errors)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("Normalize 验收通过")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"candidates: {CANDIDATES_PATH}")
    print(f"errors: {ERRORS_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

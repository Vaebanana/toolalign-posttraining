from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from profile_function_call_datasets import (
    HERMES_PATH,
    PROJECT_ROOT,
    XLAM_PATH,
    iter_json_array,
    normalize_hermes_tool_call_structure,
    parse_hermes_tool_calls,
    parse_nested_json,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "problem_samples"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

CATEGORY_DESCRIPTIONS = {
    "xlam_duplicate_query": (
        "xLAM中query与更早样本完全重复；文件只包含后续重复样本。"
    ),
    "xlam_invalid_tools_json": "xLAM的tools字段无法解析为JSON。",
    "xlam_invalid_answers_json": "xLAM的answers字段无法解析为JSON。",
    "xlam_invalid_tool_structure": "xLAM候选工具结构无效。",
    "xlam_invalid_answer_structure": "xLAM工具调用结构无效。",
    "xlam_unavailable_tool_call": "xLAM调用了候选集合之外的工具。",
    "hermes_duplicate_query": (
        "Hermes中拼接后的human query与更早样本完全重复；"
        "文件只包含后续重复样本。"
    ),
    "hermes_invalid_tools_json": "Hermes的tools字段无法解析为JSON。",
    "hermes_invalid_tool_structure": "Hermes候选工具结构无效。",
    "hermes_invalid_conversation_structure": "Hermes对话结构无效。",
    "hermes_nonstandard_tool_schema": "Hermes候选工具schema不标准。",
    "hermes_no_tool_call_tags": "Hermes assistant回答中没有tool_call标签。",
    "hermes_non_tool_assistant_text": (
        "Hermes assistant回答在移除tool_call标签后仍有普通文本。"
    ),
    "hermes_escaped_boundary_json_repaired": (
        "Hermes调用清理边界字面量转义后可按严格JSON解析。"
    ),
    "hermes_python_literal_repaired": (
        "Hermes调用需要通过Python字面量回退解析。"
    ),
    "hermes_tool_call_parse_error": "Hermes工具调用无法解析。",
    "hermes_nested_tool_name_repaired": (
        "Hermes工具名错误嵌套在arguments中，且可由候选工具确认并修复。"
    ),
    "hermes_nested_tool_name_unverifiable": (
        "Hermes工具名嵌套在arguments中，但没有候选工具可用于验证。"
    ),
    "hermes_invalid_tool_call_structure": "Hermes工具调用结构无效。",
    "hermes_unavailable_tool_call": "Hermes调用了候选集合之外的工具。",
}


class ProblemCollector:
    def __init__(self) -> None:
        self.samples: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        self.occurrences: Counter[str] = Counter()
        self.sample_keys: set[tuple[str, int]] = set()

    def add(
        self,
        category: str,
        sample_index: int,
        sample: dict[str, Any],
        occurrences: int = 1,
    ) -> None:
        self.samples[category][sample_index] = sample
        self.occurrences[category] += occurrences
        dataset = category.split("_", maxsplit=1)[0]
        self.sample_keys.add((dataset, sample_index))

    def write(self) -> dict[str, Any]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 只清理本脚本管理的已知文件，避免旧结果混入本次导出。
        for category in CATEGORY_DESCRIPTIONS:
            category_path = OUTPUT_DIR / f"{category}.jsonl"

            if category_path.exists():
                category_path.unlink()

        category_summary: dict[str, Any] = {}

        for category in CATEGORY_DESCRIPTIONS:
            indexed_samples = self.samples.get(category, {})

            if not indexed_samples:
                continue

            category_path = OUTPUT_DIR / f"{category}.jsonl"

            with category_path.open("w", encoding="utf-8") as file:
                for sample in indexed_samples.values():
                    file.write(
                        json.dumps(sample, ensure_ascii=False)
                        + "\n"
                    )

            category_summary[category] = {
                "description": CATEGORY_DESCRIPTIONS[category],
                "file": str(category_path),
                "sample_count": len(indexed_samples),
                "occurrence_count": self.occurrences[category],
            }

        summary = {
            "format": (
                "每个JSONL文件每行是一条未经字段裁剪或结构修复的"
                "完整原始样本；同一样本可属于多个问题类别。"
            ),
            "output_directory": str(OUTPUT_DIR),
            "unique_problem_samples": len(self.sample_keys),
            "categories": category_summary,
        }

        with SUMMARY_PATH.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        return summary


def collect_xlam_problems(collector: ProblemCollector) -> None:
    seen_queries: set[str] = set()

    for sample_index, sample in enumerate(iter_json_array(XLAM_PATH)):
        query = sample.get("query", "")

        if not isinstance(query, str):
            query = str(query)

        if query in seen_queries:
            collector.add("xlam_duplicate_query", sample_index, sample)
        else:
            seen_queries.add(query)

        try:
            tools = parse_nested_json(sample.get("tools", []))
        except Exception:
            collector.add("xlam_invalid_tools_json", sample_index, sample)
            tools = []

        try:
            answers = parse_nested_json(sample.get("answers", []))
        except Exception:
            collector.add("xlam_invalid_answers_json", sample_index, sample)
            answers = []

        if not isinstance(tools, list):
            collector.add("xlam_invalid_tool_structure", sample_index, sample)
            tools = []

        if not isinstance(answers, list):
            collector.add("xlam_invalid_answer_structure", sample_index, sample)
            answers = []

        available_tool_names: set[str] = set()

        for tool in tools:
            if not isinstance(tool, dict):
                collector.add(
                    "xlam_invalid_tool_structure",
                    sample_index,
                    sample,
                )
                continue

            tool_name = tool.get("name")

            if isinstance(tool_name, str):
                available_tool_names.add(tool_name)

            if not isinstance(tool.get("parameters", {}), dict):
                collector.add(
                    "xlam_invalid_tool_structure",
                    sample_index,
                    sample,
                )

        unavailable_calls = 0

        for answer in answers:
            if not isinstance(answer, dict):
                collector.add(
                    "xlam_invalid_answer_structure",
                    sample_index,
                    sample,
                )
                continue

            call_name = answer.get("name")

            if not isinstance(call_name, str):
                collector.add(
                    "xlam_invalid_answer_structure",
                    sample_index,
                    sample,
                )
            elif call_name not in available_tool_names:
                unavailable_calls += 1

            if not isinstance(answer.get("arguments"), dict):
                collector.add(
                    "xlam_invalid_answer_structure",
                    sample_index,
                    sample,
                )

        if unavailable_calls:
            collector.add(
                "xlam_unavailable_tool_call",
                sample_index,
                sample,
                unavailable_calls,
            )


def collect_hermes_problems(collector: ProblemCollector) -> None:
    seen_queries: set[str] = set()

    for sample_index, sample in enumerate(iter_json_array(HERMES_PATH)):
        try:
            tools = parse_nested_json(sample.get("tools", []))
        except Exception:
            collector.add("hermes_invalid_tools_json", sample_index, sample)
            tools = []

        if not isinstance(tools, list):
            collector.add("hermes_invalid_tool_structure", sample_index, sample)
            tools = []

        available_tool_names: set[str] = set()
        schema_invalid = False

        for tool_wrapper in tools:
            if not isinstance(tool_wrapper, dict):
                collector.add(
                    "hermes_invalid_tool_structure",
                    sample_index,
                    sample,
                )
                continue

            if tool_wrapper.get("type") != "function":
                schema_invalid = True

            function = tool_wrapper.get("function")

            if not isinstance(function, dict):
                collector.add(
                    "hermes_invalid_tool_structure",
                    sample_index,
                    sample,
                )
                continue

            tool_name = function.get("name")

            if isinstance(tool_name, str):
                available_tool_names.add(tool_name)

            parameters = function.get("parameters")

            if not isinstance(parameters, dict):
                schema_invalid = True
                continue

            if parameters.get("type") != "object":
                schema_invalid = True

            if not isinstance(parameters.get("properties", {}), dict):
                schema_invalid = True

            if not isinstance(parameters.get("required", []), list):
                schema_invalid = True

        if schema_invalid:
            collector.add("hermes_nonstandard_tool_schema", sample_index, sample)

        conversations = sample.get("conversations", [])

        if not isinstance(conversations, list):
            collector.add(
                "hermes_invalid_conversation_structure",
                sample_index,
                sample,
            )
            conversations = []

        human_messages: list[str] = []
        assistant_messages: list[str] = []

        for message in conversations:
            if not isinstance(message, dict):
                collector.add(
                    "hermes_invalid_conversation_structure",
                    sample_index,
                    sample,
                )
                continue

            role = message.get("from")
            value = message.get("value")

            if not isinstance(value, str):
                collector.add(
                    "hermes_invalid_conversation_structure",
                    sample_index,
                    sample,
                )
                continue

            if role == "human":
                human_messages.append(value)
            elif role == "gpt":
                assistant_messages.append(value)

        query = "\n".join(human_messages)

        if query in seen_queries:
            collector.add("hermes_duplicate_query", sample_index, sample)
        else:
            seen_queries.add(query)

        total_tags = 0
        non_tool_text_found = False
        normalized_calls: list[dict[str, Any]] = []

        for assistant_text in assistant_messages:
            calls, parse_stats, remaining_text = parse_hermes_tool_calls(
                assistant_text
            )
            total_tags += parse_stats["total_tags"]

            parse_categories = {
                "escaped_boundary_json": (
                    "hermes_escaped_boundary_json_repaired"
                ),
                "python_literal": "hermes_python_literal_repaired",
                "failed": "hermes_tool_call_parse_error",
            }

            for mode, category in parse_categories.items():
                if parse_stats[mode]:
                    collector.add(
                        category,
                        sample_index,
                        sample,
                        parse_stats[mode],
                    )

            if remaining_text:
                non_tool_text_found = True

            for call in calls:
                normalized_call, structure_mode = (
                    normalize_hermes_tool_call_structure(
                        call,
                        available_tool_names,
                    )
                )
                normalized_calls.append(normalized_call)

                structure_categories = {
                    "nested_name_repaired": (
                        "hermes_nested_tool_name_repaired"
                    ),
                    "nested_name_unverifiable": (
                        "hermes_nested_tool_name_unverifiable"
                    ),
                    "nested_name_not_in_candidates": (
                        "hermes_invalid_tool_call_structure"
                    ),
                    "invalid": "hermes_invalid_tool_call_structure",
                }
                category = structure_categories.get(structure_mode)

                if category:
                    collector.add(category, sample_index, sample)

        if total_tags == 0:
            collector.add("hermes_no_tool_call_tags", sample_index, sample)

        if non_tool_text_found:
            collector.add(
                "hermes_non_tool_assistant_text",
                sample_index,
                sample,
            )

        unavailable_calls = sum(
            1
            for call in normalized_calls
            if isinstance(call.get("name"), str)
            and call["name"] not in available_tool_names
        )

        if unavailable_calls:
            collector.add(
                "hermes_unavailable_tool_call",
                sample_index,
                sample,
                unavailable_calls,
            )


def main() -> None:
    for path in (XLAM_PATH, HERMES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件：{path}")

    collector = ProblemCollector()

    print("正在收集xLAM问题样本……")
    collect_xlam_problems(collector)

    print("正在收集Hermes问题样本……")
    collect_hermes_problems(collector)

    summary = collector.write()

    print(f"问题样本已写入：{OUTPUT_DIR}")
    print(f"去重后的问题样本数量：{summary['unique_problem_samples']}")

    for category, details in summary["categories"].items():
        print(
            f"{category}: "
            f"samples={details['sample_count']}, "
            f"occurrences={details['occurrence_count']}"
        )


if __name__ == "__main__":
    main()

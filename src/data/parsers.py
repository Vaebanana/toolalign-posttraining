"""xLAM与Hermes原始数据的加载和严格解析模块。

本文件只负责流水线中的Load与Parse阶段：读取JSON Array，使用
``json.loads``严格解析原始JSON字符串，并保留原始值及解析错误。
这里不执行问题审计、恢复性解析、数据修复、格式归一化或Schema校验。
"""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TypedDict


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL,
)


class JsonParseResult(TypedDict):
    success: bool
    value: Any
    error_type: str | None
    error: str | None


class HermesToolCallParse(TypedDict):
    sample_id: Any
    message_index: int
    tool_call_index: int
    raw_content: str
    parse_success: bool
    parsed_value: Any
    error_type: str | None
    error: str | None


def load_json_array(path: str | Path) -> list[Any]:
    """读取最外层必须为JSON Array的原始文件。"""
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, list):
        raise TypeError(
            f"JSON文件最外层必须是array，实际为{type(value).__name__}"
        )

    return value


def parse_json_string(value: Any) -> JsonParseResult:
    """严格使用json.loads解析字符串，不执行任何恢复或修复。"""
    if not isinstance(value, str):
        error = TypeError(
            f"待解析值必须是str，实际为{type(value).__name__}"
        )
        return {
            "success": False,
            "value": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    try:
        parsed_value = json.loads(value)
    except JSONDecodeError as error:
        return {
            "success": False,
            "value": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    return {
        "success": True,
        "value": parsed_value,
        "error_type": None,
        "error": None,
    }


def parse_xlam_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """严格解析一条xLAM样本中的JSON字符串字段。"""
    raw_answers = sample["answers"]
    raw_tools = sample["tools"]

    return {
        "id": sample["id"],
        "query": sample["query"],
        "raw_answers": raw_answers,
        "raw_tools": raw_tools,
        "answers_parse": parse_json_string(raw_answers),
        "tools_parse": parse_json_string(raw_tools),
    }


def load_xlam(path: str | Path) -> list[dict[str, Any]]:
    """读取xLAM JSON Array并严格解析每条样本。"""
    raw_samples = load_json_array(path)

    return [parse_xlam_sample(sample) for sample in raw_samples]


def parse_hermes_assistant_text(
    assistant_text: str,
    sample_id: Any,
    message_index: int,
    start_tool_call_index: int = 0,
) -> list[HermesToolCallParse]:
    """提取Hermes tool_call标签并严格解析标签原始正文。"""
    contents = TOOL_CALL_PATTERN.findall(assistant_text)
    tool_calls: list[HermesToolCallParse] = []

    for offset, content in enumerate(contents):
        parse_result = parse_json_string(content)
        tool_calls.append(
            {
                "sample_id": sample_id,
                "message_index": message_index,
                "tool_call_index": start_tool_call_index + offset,
                "raw_content": content,
                "parse_success": parse_result["success"],
                "parsed_value": parse_result["value"],
                "error_type": parse_result["error_type"],
                "error": parse_result["error"],
            }
        )

    return tool_calls


def parse_hermes_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """严格解析一条Hermes样本，不审计或修复原始问题。"""
    sample_id = sample["id"]
    raw_tools = sample["tools"]
    raw_conversations = sample["conversations"]
    tool_calls: list[HermesToolCallParse] = []

    if isinstance(raw_conversations, list):
        for message_index, message in enumerate(raw_conversations):
            if not isinstance(message, dict) or message.get("from") != "gpt":
                continue

            assistant_text = message.get("value")

            if not isinstance(assistant_text, str):
                continue

            message_tool_calls = parse_hermes_assistant_text(
                assistant_text=assistant_text,
                sample_id=sample_id,
                message_index=message_index,
                start_tool_call_index=len(tool_calls),
            )
            tool_calls.extend(message_tool_calls)

    return {
        "id": sample_id,
        "category": sample.get("category"),
        "subcategory": sample.get("subcategory"),
        "raw_tools": raw_tools,
        "raw_conversations": raw_conversations,
        "tools_parse": parse_json_string(raw_tools),
        "tool_calls": tool_calls,
    }


def load_hermes(path: str | Path) -> list[dict[str, Any]]:
    """读取Hermes JSON Array并严格解析每条样本。"""
    raw_samples = load_json_array(path)

    return [parse_hermes_sample(sample) for sample in raw_samples]

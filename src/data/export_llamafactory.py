"""Export canonical tool-use samples to LLaMA-Factory ShareGPT JSONL.

This module is deliberately only an adapter.  It maps fields, roles and JSON
representations; it does not repair, filter, sample or otherwise alter data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "llamafactory"

INPUT_FILES = {
    "posttrain_train": PROCESSED_DIR / "train.jsonl",
    "posttrain_dev_seen": PROCESSED_DIR / "dev_seen.jsonl",
}
OUTPUT_FILES = {
    "posttrain_train": OUTPUT_DIR / "train.jsonl",
    "posttrain_dev_seen": OUTPUT_DIR / "dev_seen.jsonl",
}

ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "tool": "observation",
}


class ExportError(ValueError):
    """A canonical sample cannot be mapped without changing its meaning."""


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{field} must be an object")
    return value


def _require_array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExportError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ExportError(f"{field} must be a string")
    return value


def _serialize_tool_calls(tool_calls: list[Any], sample_id: str) -> str:
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(tool_calls):
        call = _require_object(
            raw_call,
            f"{sample_id}.assistant.tool_calls[{index}]",
        )
        if call.get("type") != "function":
            raise ExportError(
                f"{sample_id}.assistant.tool_calls[{index}].type "
                "must be 'function'"
            )
        function = _require_object(
            call.get("function"),
            f"{sample_id}.assistant.tool_calls[{index}].function",
        )
        calls.append(
            {
                "name": _require_string(
                    function.get("name"),
                    f"{sample_id}.assistant.tool_calls[{index}]"
                    ".function.name",
                ),
                "arguments": _require_object(
                    function.get("arguments"),
                    f"{sample_id}.assistant.tool_calls[{index}]"
                    ".function.arguments",
                ),
            }
        )

    value: dict[str, Any] | list[dict[str, Any]]
    value = calls[0] if len(calls) == 1 else calls
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def export_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Map one canonical sample to LLaMA-Factory's ShareGPT schema."""
    sample_id = _require_string(sample.get("sample_id"), "sample_id")
    messages = _require_array(sample.get("messages"), f"{sample_id}.messages")
    tools = _require_array(sample.get("tools"), f"{sample_id}.tools")
    assistant = _require_object(
        sample.get("assistant"),
        f"{sample_id}.assistant",
    )

    system_messages: list[str] = []
    conversations: list[dict[str, str]] = []
    for index, raw_message in enumerate(messages):
        message = _require_object(
            raw_message,
            f"{sample_id}.messages[{index}]",
        )
        role = _require_string(
            message.get("role"),
            f"{sample_id}.messages[{index}].role",
        )
        content = _require_string(
            message.get("content"),
            f"{sample_id}.messages[{index}].content",
        )
        if role == "system":
            system_messages.append(content)
            continue
        if role not in ROLE_MAP:
            raise ExportError(
                f"{sample_id}.messages[{index}].role is unsupported: {role!r}"
            )
        conversations.append({"from": ROLE_MAP[role], "value": content})

    if len(system_messages) != 1:
        raise ExportError(
            f"{sample_id} must contain exactly one system message; "
            f"got {len(system_messages)}"
        )

    tool_calls = _require_array(
        assistant.get("tool_calls"),
        f"{sample_id}.assistant.tool_calls",
    )
    content = assistant.get("content")
    if tool_calls:
        if content is not None:
            raise ExportError(
                f"{sample_id}.assistant cannot contain both content and tool_calls"
            )
        conversations.append(
            {
                "from": "function_call",
                "value": _serialize_tool_calls(tool_calls, sample_id),
            }
        )
    else:
        conversations.append(
            {
                "from": "gpt",
                "value": _require_string(
                    content,
                    f"{sample_id}.assistant.content",
                ),
            }
        )

    return {
        "conversations": conversations,
        "system": system_messages[0],
        "tools": json.dumps(
            tools,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExportError(f"{path}:{line_number}: invalid JSON") from error
            yield line_number, _require_object(value, f"{path}:{line_number}")


def export_file(input_path: Path, output_path: Path) -> int:
    """Export every input record, preserving order and cardinality."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for line_number, sample in _read_jsonl(input_path):
            try:
                exported = export_sample(sample)
            except ExportError as error:
                raise ExportError(f"{input_path}:{line_number}: {error}") from error
            output.write(json.dumps(exported, ensure_ascii=False) + "\n")
            count += 1
    return count


def _dataset_info() -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": OUTPUT_FILES[dataset_name].name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
        }
        for dataset_name in INPUT_FILES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical train/dev_seen data for LLaMA-Factory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: data/llamafactory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_files = {
        name: output_dir / path.name for name, path in OUTPUT_FILES.items()
    }

    counts: dict[str, int] = {}
    for dataset_name, input_path in INPUT_FILES.items():
        counts[dataset_name] = export_file(
            input_path,
            output_files[dataset_name],
        )

    dataset_info = _dataset_info()
    for dataset_name, output_path in output_files.items():
        dataset_info[dataset_name]["file_name"] = output_path.name
    info_path = output_dir / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("LLaMA-Factory export complete")
    for dataset_name, count in counts.items():
        print(f"{dataset_name}: {count}")
    print(f"dataset_info: {info_path}")


if __name__ == "__main__":
    main()

"""xLAM与Hermes严格解析器的全量验收脚本。

本文件调用``parsers.py``解析完整原始数据集，并汇总样本数、严格解析
成功数和失败数，用于确认解析器准确反映原始数据状态。它只做结果计数
和一致性检查，不执行问题审计或数据修复。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parsers import load_hermes, load_xlam


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


def count_failed(
    samples: list[dict[str, Any]],
    parse_field: str,
) -> int:
    return sum(
        not sample[parse_field]["success"]
        for sample in samples
    )


def main() -> None:
    print("正在执行xLAM严格解析……")
    xlam_samples = load_xlam(XLAM_PATH)

    print("xLAM total:", len(xlam_samples))
    print(
        "xLAM answers parse failed:",
        count_failed(xlam_samples, "answers_parse"),
    )
    print(
        "xLAM tools parse failed:",
        count_failed(xlam_samples, "tools_parse"),
    )

    # 两个完整数据集不需要同时常驻内存。
    del xlam_samples

    print("正在执行Hermes严格解析……")
    hermes_samples = load_hermes(HERMES_PATH)

    tools_failed = count_failed(hermes_samples, "tools_parse")
    total_tags = 0
    parsed_tags = 0
    failed_tags = 0
    samples_without_tags = 0

    for sample in hermes_samples:
        sample_tag_count = len(sample["tool_calls"])
        total_tags += sample_tag_count

        for tool_call in sample["tool_calls"]:
            if tool_call["parse_success"]:
                parsed_tags += 1
            else:
                failed_tags += 1

        if sample_tag_count == 0:
            samples_without_tags += 1

    if parsed_tags + failed_tags != total_tags:
        raise RuntimeError("Hermes严格解析计数不一致")

    print("Hermes total:", len(hermes_samples))
    print("Hermes tools parse failed:", tools_failed)
    print("Hermes tool_call tags:", total_tags)
    print("Hermes tool_call strict parse success:", parsed_tags)
    print("Hermes tool_call strict parse failed:", failed_tags)
    print("Hermes samples without tool_call tags:", samples_without_tags)


if __name__ == "__main__":
    main()

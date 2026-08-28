"""Validate the adapter and LLaMA-Factory's real SFT interpretation.

The framework sanity check intentionally runs on six representative records:
two single-call, two parallel-call and two text-only assistant targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from export_llamafactory import (
    INPUT_FILES,
    OUTPUT_DIR,
    OUTPUT_FILES,
    PROJECT_ROOT,
    export_sample,
)


DEFAULT_TOKENIZER_DIR = (
    PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507-tokenizer"
)
SANITY_DIR = PROJECT_ROOT / "outputs" / "llamafactory_sanity"
IGNORE_INDEX = -100
SANITY_PER_TYPE = 2
EXPECTED_DATASET_COUNTS = {
    "posttrain_train": 50_575,
    "posttrain_dev_seen": 2_802,
}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise AssertionError(f"{path}:{line_number}: row is not an object")
            yield value


def _sample_type(sample: dict[str, Any]) -> str:
    tool_calls = sample["assistant"]["tool_calls"]
    if not tool_calls:
        return "text"
    if len(tool_calls) == 1:
        return "single"
    return "multi"


def _validate_exported_files() -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset_name, canonical_path in INPUT_FILES.items():
        exported_path = OUTPUT_FILES[dataset_name]
        canonical_rows = _read_jsonl(canonical_path)
        exported_rows = _read_jsonl(exported_path)
        count = 0
        while True:
            canonical = next(canonical_rows, None)
            exported = next(exported_rows, None)
            if canonical is None or exported is None:
                if canonical is not None or exported is not None:
                    raise AssertionError(
                        f"cardinality mismatch: {canonical_path} vs {exported_path}"
                    )
                break

            expected = export_sample(canonical)
            if exported != expected:
                raise AssertionError(
                    f"adapter output mismatch for {canonical['sample_id']}"
                )
            if set(exported) != {"conversations", "system", "tools"}:
                raise AssertionError(
                    f"unexpected exported fields for {canonical['sample_id']}"
                )
            if json.loads(exported["tools"]) != canonical["tools"]:
                raise AssertionError(
                    f"tool schema changed for {canonical['sample_id']}"
                )
            count += 1

        expected_count = EXPECTED_DATASET_COUNTS[dataset_name]
        if count != expected_count:
            raise AssertionError(
                f"{dataset_name}: expected {expected_count}, got {count}"
            )
        counts[dataset_name] = count

    with (OUTPUT_DIR / "dataset_info.json").open("r", encoding="utf-8") as file:
        dataset_info = json.load(file)
    if set(dataset_info) != set(INPUT_FILES):
        raise AssertionError("dataset_info.json has unexpected dataset names")
    for dataset_name in INPUT_FILES:
        attributes = dataset_info[dataset_name]
        if attributes.get("formatting") != "sharegpt":
            raise AssertionError(f"{dataset_name}: formatting is not sharegpt")
        if attributes.get("columns") != {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
        }:
            raise AssertionError(f"{dataset_name}: column mapping is incorrect")

    return counts


def _select_representatives() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates: dict[
        str,
        list[tuple[int, dict[str, Any], dict[str, Any]]],
    ] = {
        "single": [],
        "multi": [],
        "text": [],
    }
    canonical_rows = _read_jsonl(INPUT_FILES["posttrain_train"])
    exported_rows = _read_jsonl(OUTPUT_FILES["posttrain_train"])
    for canonical, exported in zip(canonical_rows, exported_rows, strict=True):
        kind = _sample_type(canonical)
        encoded_size = len(json.dumps(exported, ensure_ascii=False))
        candidates[kind].append((encoded_size, canonical, exported))

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for kind in ("single", "multi", "text"):
        if len(candidates[kind]) < SANITY_PER_TYPE:
            raise AssertionError(f"not enough {kind} samples for sanity check")
        candidates[kind].sort(key=lambda item: (item[0], item[1]["sample_id"]))
        selected.extend(
            (canonical, exported)
            for _, canonical, exported in candidates[kind][:SANITY_PER_TYPE]
        )
    return selected


def _write_sanity_dataset(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    with (SANITY_DIR / "sanity.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        for _, exported in selected:
            file.write(json.dumps(exported, ensure_ascii=False) + "\n")

    dataset_info = {
        "posttrain_sanity": {
            "file_name": "sanity.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
        }
    }
    with (SANITY_DIR / "dataset_info.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _load_with_llamafactory(tokenizer_dir: Path) -> tuple[Any, Any]:
    # These imports are intentionally local: failure proves that the selected
    # conda environment does not contain the real LLaMA-Factory stack.
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import (
        DataArguments,
        ModelArguments,
        TrainingArguments,
    )
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(model_name_or_path=str(tokenizer_dir))
    data_args = DataArguments(
        template="qwen3_nothink",
        dataset="posttrain_sanity",
        dataset_dir=str(SANITY_DIR),
        cutoff_len=4096,
        train_on_prompt=False,
        overwrite_cache=True,
        preprocessing_batch_size=6,
        preprocessing_num_workers=1,
    )
    training_args = TrainingArguments(
        output_dir=str(SANITY_DIR / "trainer_output"),
        do_train=True,
        per_device_train_batch_size=1,
        report_to="none",
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
    )
    return tokenizer, dataset_module["train_dataset"]


def _validate_tokenized_sample(
    canonical: dict[str, Any],
    tokenized: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    if len(input_ids) != len(labels):
        raise AssertionError(f"{canonical['sample_id']}: input/label length mismatch")

    supervised_positions = [
        index for index, label in enumerate(labels) if label != IGNORE_INDEX
    ]
    if not supervised_positions:
        raise AssertionError(f"{canonical['sample_id']}: no supervised labels")
    first_target = supervised_positions[0]
    if first_target == 0 or any(label != IGNORE_INDEX for label in labels[:first_target]):
        raise AssertionError(f"{canonical['sample_id']}: prompt labels are not masked")
    if any(labels[index] != input_ids[index] for index in supervised_positions):
        raise AssertionError(f"{canonical['sample_id']}: target labels differ from input ids")

    prompt_text = tokenizer.decode(input_ids[:first_target], skip_special_tokens=False)
    target_ids = [labels[index] for index in supervised_positions]
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    user_texts = [
        message["content"]
        for message in canonical["messages"]
        if message["role"] == "user"
    ]
    if not all(text in prompt_text for text in user_texts):
        raise AssertionError(f"{canonical['sample_id']}: user text missing from prompt")
    tool_names = [tool["function"]["name"] for tool in canonical["tools"]]
    if not all(name in prompt_text for name in tool_names):
        raise AssertionError(f"{canonical['sample_id']}: tool schema missing from prompt")
    if "<tools>" not in prompt_text or "</tools>" not in prompt_text:
        raise AssertionError(f"{canonical['sample_id']}: Qwen tools block missing")

    calls = canonical["assistant"]["tool_calls"]
    kind = _sample_type(canonical)
    if calls:
        if target_text.count("<tool_call>") != len(calls):
            raise AssertionError(
                f"{canonical['sample_id']}: wrong number of supervised tool calls"
            )
        for call in calls:
            if call["function"]["name"] not in target_text:
                raise AssertionError(
                    f"{canonical['sample_id']}: tool name missing from target"
                )
    else:
        content = canonical["assistant"]["content"]
        if content not in target_text or "<tool_call>" in target_text:
            raise AssertionError(
                f"{canonical['sample_id']}: text response target is incorrect"
            )

    return {
        "sample_id": canonical["sample_id"],
        "type": kind,
        "input_tokens": len(input_ids),
        "masked_prompt_tokens": first_target,
        "supervised_tokens": len(supervised_positions),
        "expected_tool_calls": len(calls),
        "target_preview": target_text[:500],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check exported data with real LLaMA-Factory qwen3_nothink preprocessing."
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Local Qwen3 tokenizer directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir.resolve()
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(
            f"Qwen3 tokenizer not found at {tokenizer_dir}. "
            "Download tokenizer files before running this check."
        )

    export_counts = _validate_exported_files()
    selected = _select_representatives()
    _write_sanity_dataset(selected)
    tokenizer, tokenized_dataset = _load_with_llamafactory(tokenizer_dir)
    if tokenized_dataset is None or len(tokenized_dataset) != len(selected):
        raise AssertionError(
            f"LLaMA-Factory loaded {0 if tokenized_dataset is None else len(tokenized_dataset)} "
            f"samples; expected {len(selected)}"
        )

    records = [
        _validate_tokenized_sample(canonical, tokenized_dataset[index], tokenizer)
        for index, (canonical, _) in enumerate(selected)
    ]
    type_counts = Counter(record["type"] for record in records)
    if type_counts != Counter({"single": 2, "multi": 2, "text": 2}):
        raise AssertionError(f"unexpected representative types: {type_counts}")

    summary = {
        "status": "passed",
        "llamafactory_template": "qwen3_nothink",
        "tokenizer": str(tokenizer_dir),
        "export_counts": export_counts,
        "sanity_samples": len(records),
        "sample_type_counts": dict(type_counts),
        "train_on_prompt": False,
        "records": records,
    }
    summary_path = SANITY_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("LLaMA-Factory qwen3_nothink sanity check passed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

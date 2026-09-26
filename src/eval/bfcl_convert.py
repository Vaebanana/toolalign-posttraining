"""Convert saved raw model output to official BFCL v4 result JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.eval.bfcl_common import (
    BFCLAdapterError,
    DEFAULT_CATEGORIES,
    DEFAULT_RAW_DIR,
    DEFAULT_REGISTRY_NAME,
    DEFAULT_RESULT_DIR,
    dataset_path,
    load_jsonl,
    official_result_path,
    parse_categories,
    raw_prediction_path,
    require_unique_ids,
    resolve_bfcl_data_dir,
)


def _optional_number(value: Any, field: str, source: Path, line: int) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BFCLAdapterError(f"{source}:{line}: {field} must be numeric")
    return value


def to_official_record(
    prediction: dict[str, Any],
    *,
    category: str,
    source: Path,
    line: int,
) -> dict[str, Any]:
    entry_id = prediction.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise BFCLAdapterError(f"{source}:{line}: id must be a non-empty string")
    if prediction.get("test_category") != category:
        raise BFCLAdapterError(
            f"{source}:{line}: test_category must be {category!r}"
        )
    raw_prediction = prediction.get("raw_prediction")
    if not isinstance(raw_prediction, str):
        raise BFCLAdapterError(f"{source}:{line}: raw_prediction must be a string")
    generation = prediction.get("generation", {})
    if not isinstance(generation, dict):
        raise BFCLAdapterError(f"{source}:{line}: generation must be an object")

    record: dict[str, Any] = {"id": entry_id, "result": raw_prediction}
    metadata_fields = (
        ("prompt_tokens", "input_token_count"),
        ("response_tokens", "output_token_count"),
        ("latency_seconds", "latency"),
    )
    for raw_field, official_field in metadata_fields:
        value = _optional_number(generation.get(raw_field), raw_field, source, line)
        if value is not None:
            record[official_field] = value
    return record


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_category(
    *,
    category: str,
    data_dir: Path,
    raw_dir: Path,
    result_dir: Path,
    registry_name: str,
    allow_partial: bool = False,
) -> Path:
    """Validate and convert one category without altering model text."""
    source_data = dataset_path(data_dir, category)
    dataset_entries = load_jsonl(source_data)
    dataset_ids = require_unique_ids(dataset_entries, source=source_data)
    dataset_id_set = set(dataset_ids)

    raw_path = raw_prediction_path(raw_dir, category)
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw BFCL predictions not found: {raw_path}")
    predictions = load_jsonl(raw_path)
    prediction_ids = require_unique_ids(predictions, source=raw_path)
    prediction_id_set = set(prediction_ids)
    unknown = prediction_id_set - dataset_id_set
    missing = dataset_id_set - prediction_id_set
    if unknown:
        preview = ", ".join(repr(item) for item in sorted(unknown)[:5])
        raise BFCLAdapterError(f"{raw_path}: unknown BFCL ids: {preview}")
    if missing and not allow_partial:
        raise BFCLAdapterError(
            f"{raw_path}: incomplete category {category}: "
            f"{len(prediction_ids)}/{len(dataset_ids)} predictions; rerun inference "
            "or use --allow-partial only for a smoke evaluation"
        )

    by_id = {prediction["id"]: (line, prediction) for line, prediction in enumerate(predictions, 1)}
    official_records = [
        to_official_record(
            by_id[entry_id][1],
            category=category,
            source=raw_path,
            line=by_id[entry_id][0],
        )
        for entry_id in dataset_ids
        if entry_id in by_id
    ]
    output_path = official_result_path(result_dir, registry_name, category)
    _atomic_write_jsonl(output_path, official_records)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Qwen3 predictions into official BFCL v4 result files."
    )
    parser.add_argument("--bfcl-data-dir", type=Path, default=None)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--registry-name", default=DEFAULT_REGISTRY_NAME)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete result files for BFCL --partial-eval smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.categories)
    data_dir = resolve_bfcl_data_dir(args.bfcl_data_dir)
    if not args.registry_name or any(char in args.registry_name for char in "_/\\"):
        raise BFCLAdapterError(
            "registry-name must be non-empty and contain no underscore or slash; "
            "BFCL rewrites those characters while resolving model handlers"
        )
    for category in categories:
        output = convert_category(
            category=category,
            data_dir=data_dir,
            raw_dir=args.raw_dir.resolve(),
            result_dir=args.result_dir.resolve(),
            registry_name=args.registry_name,
            allow_partial=args.allow_partial,
        )
        count = len(load_jsonl(output))
        print(f"converted: {category} {count} -> {output}")
    print("BFCL result conversion complete")


if __name__ == "__main__":
    main()

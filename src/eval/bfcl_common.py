"""Shared, dependency-light contracts for the local BFCL v4 pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PREFIX = "BFCL_v4"
DEFAULT_CATEGORIES = (
    "live_simple",
    "live_multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
SUPPORTED_CATEGORIES = frozenset(DEFAULT_CATEGORIES)
CATEGORY_GROUP = {
    "live_simple": "live",
    "live_multiple": "live",
    "parallel": "non_live",
    "parallel_multiple": "non_live",
    "irrelevance": "non_live",
}

DEFAULT_RUN_NAME = "dpo_v1"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "qwen3_4b_sft_v1_merged"
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "outputs" / "train" / "qwen3_4b_qlora_dpo_v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "bfcl" / DEFAULT_RUN_NAME
DEFAULT_RAW_DIR = DEFAULT_RUN_ROOT / "raw"
DEFAULT_RESULT_DIR = DEFAULT_RUN_ROOT / "result"
DEFAULT_SCORE_DIR = DEFAULT_RUN_ROOT / "score"
DEFAULT_REGISTRY_NAME = "toolalign-qwen3-4b-dpo-v1-FC"
DEFAULT_DISPLAY_NAME = "ToolAlign Qwen3-4B DPO v1 (FC)"


class BFCLAdapterError(ValueError):
    """The requested BFCL adaptation would be ambiguous or lossy."""


def parse_categories(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse, validate, and de-duplicate concrete BFCL category names."""
    raw_values = [value] if isinstance(value, str) else list(value)
    categories: list[str] = []
    for raw_value in raw_values:
        categories.extend(item.strip() for item in raw_value.split(","))
    categories = [item for item in categories if item]
    if not categories:
        raise BFCLAdapterError("at least one BFCL category is required")

    unknown = sorted(set(categories) - SUPPORTED_CATEGORIES)
    if unknown:
        raise BFCLAdapterError(
            "unsupported BFCL categories: "
            + ", ".join(unknown)
            + "; supported: "
            + ", ".join(DEFAULT_CATEGORIES)
        )
    return tuple(dict.fromkeys(categories))


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects and fail with a source location on bad input."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise BFCLAdapterError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BFCLAdapterError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise BFCLAdapterError(
                    f"{path}:{line_number}: JSONL entry must be an object"
                )
            yield line_number, value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [entry for _, entry in read_jsonl(path)]


def dataset_path(data_dir: Path, category: str) -> Path:
    return data_dir / f"{VERSION_PREFIX}_{category}.json"


def raw_prediction_path(raw_dir: Path, category: str) -> Path:
    return raw_dir / f"{VERSION_PREFIX}_{category}_predictions.jsonl"


def official_result_path(
    result_dir: Path,
    registry_name: str,
    category: str,
) -> Path:
    return (
        result_dir
        / registry_name
        / CATEGORY_GROUP[category]
        / f"{VERSION_PREFIX}_{category}_result.json"
    )


def resolve_bfcl_data_dir(explicit: Path | None) -> Path:
    """Locate package data without importing BFCL into the PyTorch env."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_value = os.getenv("BFCL_DATA_DIR")
    if env_value:
        candidates.append(Path(env_value))
    # Convenient for this workspace layout while keeping --bfcl-data-dir the
    # portable, authoritative option.
    candidates.append(
        PROJECT_ROOT.parent
        / "bfcl"
        / "gorilla"
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if all(dataset_path(resolved, category).is_file() for category in DEFAULT_CATEGORIES):
            return resolved

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "BFCL v4 package data was not found. Pass --bfcl-data-dir or set "
        f"BFCL_DATA_DIR. Checked: {checked}"
    )


def require_unique_ids(
    entries: Iterable[dict[str, Any]],
    *,
    source: Path,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise BFCLAdapterError(f"{source}:{index}: id must be a non-empty string")
        if entry_id in seen:
            raise BFCLAdapterError(f"{source}:{index}: duplicate id {entry_id!r}")
        seen.add(entry_id)
        ids.append(entry_id)
    return ids

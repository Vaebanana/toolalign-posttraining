"""Build an auditable, deterministic subset of LLaMA-Factory JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "llamafactory" / "train.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "llamafactory" / "train_sft_v1_10k.jsonl"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "llamafactory"
    / "train_sft_v1_10k.manifest.json"
)
DEFAULT_SAMPLE_COUNT = 10_000
DEFAULT_SEED = 42
SAMPLING_METHOD = "python_random_sample_without_replacement_v1"


class SubsetError(ValueError):
    """The requested subset cannot be built safely."""


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _inspect_jsonl(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise SubsetError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SubsetError(
                    f"{path}:{line_number}: invalid UTF-8 JSON object"
                ) from error
            if not isinstance(value, dict):
                raise SubsetError(f"{path}:{line_number}: expected JSON object")
            count += 1
    return count, digest.hexdigest()


def select_line_indices(total_count: int, sample_count: int, seed: int) -> list[int]:
    """Return sorted zero-based indices sampled uniformly without replacement."""
    if sample_count <= 0:
        raise SubsetError("sample_count must be positive")
    if sample_count > total_count:
        raise SubsetError(
            f"sample_count={sample_count} exceeds source_count={total_count}"
        )
    return sorted(random.Random(seed).sample(range(total_count), sample_count))


def _write_subset(
    input_path: Path,
    output_path: Path,
    selected_indices: list[int],
) -> str:
    if input_path.resolve() == output_path.resolve():
        raise SubsetError("input and output paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = set(selected_indices)
    output_digest = hashlib.sha256()
    written = 0
    with input_path.open("rb") as source, output_path.open("wb") as output:
        for index, raw_line in enumerate(source):
            if index not in selected:
                continue
            normalized_line = raw_line.rstrip(b"\r\n") + b"\n"
            output.write(normalized_line)
            output_digest.update(normalized_line)
            written += 1

    if written != len(selected_indices):
        raise SubsetError(
            f"selected {len(selected_indices)} records but wrote {written}"
        )
    return output_digest.hexdigest()


def build_subset(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build a subset and return the manifest written beside it."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if not input_path.is_file():
        raise SubsetError(f"source file does not exist: {input_path}")

    source_count, source_sha256 = _inspect_jsonl(input_path)
    selected_indices = select_line_indices(source_count, sample_count, seed)
    output_sha256 = _write_subset(input_path, output_path, selected_indices)

    index_bytes = "\n".join(str(index + 1) for index in selected_indices).encode()
    manifest: dict[str, Any] = {
        "manifest_version": "toolalign-sft-subset-v1",
        "source": "train",
        "source_file": _display_path(input_path),
        "source_sample_count": source_count,
        "source_sha256": source_sha256,
        "seed": seed,
        "sample_count": sample_count,
        "sampling_method": SAMPLING_METHOD,
        "output_order": "source_line_order",
        "selected_line_numbers_base": 1,
        "selected_line_numbers_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "selected_line_numbers": [index + 1 for index in selected_indices],
        "output_file": _display_path(output_path),
        "output_sha256": output_sha256,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic 10k LLaMA-Factory SFT-v1 subset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_subset(
        args.input,
        args.output,
        args.manifest,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print("SFT-v1 subset ready")
    print(f"source: {manifest['source']}")
    print(f"seed: {manifest['seed']}")
    print(f"samples: {manifest['sample_count']}")
    print(f"method: {manifest['sampling_method']}")
    print(f"output: {Path(args.output).resolve()}")
    print(f"manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()

"""Build an auditable train-side pool for SFT-v1 hard-negative mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from .export_llamafactory import export_sample
except ImportError:  # Support ``python src/data/build_dpo_mining_pool.py``.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.export_llamafactory import export_sample


DEFAULT_CANONICAL_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_LLAMA_TRAIN = PROJECT_ROOT / "data" / "llamafactory" / "train.jsonl"
DEFAULT_SFT_MANIFEST = (
    PROJECT_ROOT / "data" / "llamafactory" / "train_sft_v1_10k.manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dpo" / "mining_pool_v1_5k.jsonl"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "dpo" / "mining_pool_v1_5k.manifest.json"
)
DEFAULT_SAMPLE_COUNT = 5_000
DEFAULT_SEED = 43
SAMPLING_METHOD = "python_random_sample_without_replacement_from_eligible_v1"


class MiningPoolError(ValueError):
    """The requested mining pool cannot be built safely."""


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json_object(raw: bytes, path: Path, line_number: int) -> dict[str, Any]:
    if not raw.strip():
        raise MiningPoolError(f"{path}:{line_number}: blank JSONL line")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MiningPoolError(
            f"{path}:{line_number}: invalid UTF-8 JSON object"
        ) from error
    if not isinstance(value, dict):
        raise MiningPoolError(f"{path}:{line_number}: expected JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: list[str | int]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_sft_manifest(path: Path, llama_train_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SFT subset manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MiningPoolError(f"{path}: invalid JSON") from error
    if not isinstance(manifest, dict):
        raise MiningPoolError(f"{path}: manifest must be an object")
    if manifest.get("manifest_version") != "toolalign-sft-subset-v1":
        raise MiningPoolError(f"{path}: unsupported SFT manifest version")
    line_numbers = manifest.get("selected_line_numbers")
    if (
        not isinstance(line_numbers, list)
        or not line_numbers
        or not all(isinstance(value, int) and value > 0 for value in line_numbers)
        or len(line_numbers) != len(set(line_numbers))
    ):
        raise MiningPoolError(f"{path}: invalid selected_line_numbers")
    if line_numbers != sorted(line_numbers):
        raise MiningPoolError(f"{path}: selected_line_numbers must be sorted")
    if manifest.get("sample_count") != len(line_numbers):
        raise MiningPoolError(f"{path}: SFT sample count does not match line numbers")
    expected_line_hash = _sequence_sha256(line_numbers)
    if manifest.get("selected_line_numbers_sha256") != expected_line_hash:
        raise MiningPoolError(f"{path}: selected line-number hash mismatch")
    actual_llama_hash = _sha256_file(llama_train_path)
    if manifest.get("source_sha256") != actual_llama_hash:
        raise MiningPoolError(
            f"{path}: LLaMA-Factory train hash no longer matches SFT source"
        )
    return manifest


def _inspect_aligned_sources(
    canonical_path: Path,
    llama_path: Path,
    excluded_lines: set[int],
) -> tuple[str, str, list[tuple[int, str]], list[str], dict[int, str]]:
    """Verify full export alignment and return eligible/excluded identities."""
    canonical_digest = hashlib.sha256()
    llama_digest = hashlib.sha256()
    eligible: list[tuple[int, str]] = []
    excluded_ids: list[str] = []
    seen_ids: set[str] = set()
    sample_ids_by_line: dict[int, str] = {}

    with canonical_path.open("rb") as canonical_file, llama_path.open("rb") as llama_file:
        pairs = zip_longest(canonical_file, llama_file)
        for line_number, pair in enumerate(pairs, start=1):
            canonical_raw, llama_raw = pair
            if canonical_raw is None or llama_raw is None:
                raise MiningPoolError(
                    "canonical and LLaMA-Factory train cardinalities differ"
                )
            canonical_digest.update(canonical_raw)
            llama_digest.update(llama_raw)
            canonical = _load_json_object(canonical_raw, canonical_path, line_number)
            llama = _load_json_object(llama_raw, llama_path, line_number)
            try:
                exported = export_sample(canonical)
            except Exception as error:
                raise MiningPoolError(
                    f"{canonical_path}:{line_number}: cannot reproduce export"
                ) from error
            if exported != llama:
                raise MiningPoolError(
                    f"line {line_number}: canonical and LLaMA-Factory rows are misaligned"
                )
            sample_id = canonical.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise MiningPoolError(
                    f"{canonical_path}:{line_number}: invalid sample_id"
                )
            if sample_id in seen_ids:
                raise MiningPoolError(
                    f"{canonical_path}:{line_number}: duplicate sample_id {sample_id!r}"
                )
            seen_ids.add(sample_id)
            sample_ids_by_line[line_number] = sample_id
            if line_number in excluded_lines:
                excluded_ids.append(sample_id)
            else:
                eligible.append((line_number, sample_id))

    missing_exclusions = excluded_lines - {
        line_number for line_number in range(1, len(eligible) + len(excluded_ids) + 1)
    }
    if missing_exclusions:
        raise MiningPoolError(
            f"SFT manifest line numbers exceed source: {sorted(missing_exclusions)[:5]}"
        )
    if len(excluded_ids) != len(excluded_lines):
        raise MiningPoolError("not every SFT line number resolved to a sample ID")
    return (
        canonical_digest.hexdigest(),
        llama_digest.hexdigest(),
        eligible,
        excluded_ids,
        sample_ids_by_line,
    )


def _load_previous_mining_manifests(
    paths: list[Path],
) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"previous mining manifest not found: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise MiningPoolError(f"{path}: invalid JSON") from error
        if not isinstance(manifest, dict):
            raise MiningPoolError(f"{path}: manifest must be an object")
        if manifest.get("manifest_version") != "toolalign-dpo-mining-pool-v1":
            raise MiningPoolError(f"{path}: unsupported mining manifest version")
        line_numbers = manifest.get("selected_source_line_numbers")
        sample_ids = manifest.get("sample_ids")
        if (
            not isinstance(line_numbers, list)
            or not line_numbers
            or not all(isinstance(value, int) and value > 0 for value in line_numbers)
            or line_numbers != sorted(line_numbers)
            or len(line_numbers) != len(set(line_numbers))
        ):
            raise MiningPoolError(f"{path}: invalid selected source line numbers")
        if (
            not isinstance(sample_ids, list)
            or not all(isinstance(value, str) and value for value in sample_ids)
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise MiningPoolError(f"{path}: invalid sample_ids")
        if manifest.get("sample_count") != len(line_numbers) or len(
            line_numbers
        ) != len(sample_ids):
            raise MiningPoolError(f"{path}: mining sample counts differ")
        if manifest.get("selected_source_line_numbers_sha256") != _sequence_sha256(
            line_numbers
        ):
            raise MiningPoolError(f"{path}: selected line-number hash mismatch")
        if manifest.get("sample_ids_sha256") != _sequence_sha256(sample_ids):
            raise MiningPoolError(f"{path}: sample ID hash mismatch")
        loaded.append((path, manifest))
    return loaded


def _write_selected_rows(
    canonical_path: Path,
    output_path: Path,
    selected_lines: set[int],
) -> str:
    if canonical_path.resolve() == output_path.resolve():
        raise MiningPoolError("source and output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with canonical_path.open("rb") as source, output_path.open("wb") as output:
        for line_number, raw_line in enumerate(source, start=1):
            if line_number not in selected_lines:
                continue
            normalized = raw_line.rstrip(b"\r\n") + b"\n"
            output.write(normalized)
            digest.update(normalized)
            written += 1
    if written != len(selected_lines):
        raise MiningPoolError(
            f"selected {len(selected_lines)} records but wrote {written}"
        )
    return digest.hexdigest()


def build_mining_pool(
    canonical_train_path: Path,
    llama_train_path: Path,
    sft_manifest_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    previous_mining_manifest_paths: list[Path] | None = None,
    candidate_prescreen_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Exclude SFT-v1 examples and uniformly sample a canonical mining pool."""
    paths = [canonical_train_path, llama_train_path, sft_manifest_path]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    sft_manifest = _load_sft_manifest(sft_manifest_path, llama_train_path)
    previous_mining_manifest_paths = previous_mining_manifest_paths or []
    previous_manifests = _load_previous_mining_manifests(
        previous_mining_manifest_paths
    )
    sft_line_numbers = set(sft_manifest["selected_line_numbers"])
    previous_line_numbers: set[int] = set()
    for path, manifest in previous_manifests:
        selected_lines = set(manifest["selected_source_line_numbers"])
        overlap = selected_lines & (sft_line_numbers | previous_line_numbers)
        if overlap:
            raise MiningPoolError(
                f"{path}: exclusion overlaps an earlier exclusion at lines "
                f"{sorted(overlap)[:5]}"
            )
        previous_line_numbers.update(selected_lines)
    excluded_line_numbers = sft_line_numbers | previous_line_numbers
    (
        canonical_sha256,
        llama_sha256,
        eligible,
        excluded_ids,
        sample_ids_by_line,
    ) = _inspect_aligned_sources(
        canonical_train_path,
        llama_train_path,
        excluded_line_numbers,
    )
    source_count = len(eligible) + len(excluded_ids)
    if source_count != sft_manifest.get("source_sample_count"):
        raise MiningPoolError("source count differs from SFT manifest")
    eligible_before_candidate_filter = len(eligible)
    candidate_prescreen: dict[str, Any] | None = None
    if candidate_prescreen_manifest_path is not None:
        if not candidate_prescreen_manifest_path.is_file():
            raise FileNotFoundError(
                "candidate prescreen manifest not found: "
                f"{candidate_prescreen_manifest_path}"
            )
        prescreen = json.loads(
            candidate_prescreen_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(prescreen, dict):
            raise MiningPoolError("candidate prescreen manifest must be an object")
        if prescreen.get("manifest_version") != (
            "toolalign-dpo-groundability-rule-triage-v1"
        ):
            raise MiningPoolError("unsupported candidate prescreen manifest version")
        if prescreen.get("status") != "rule_triage_complete_no_rollout":
            raise MiningPoolError("candidate prescreen is not complete")
        operations = prescreen.get("operations")
        if not isinstance(operations, dict) or any(operations.values()):
            raise MiningPoolError("candidate prescreen unexpectedly records mutations")
        source = prescreen.get("source", {})
        if source.get("canonical_sha256") != canonical_sha256:
            raise MiningPoolError("candidate prescreen canonical source hash mismatch")
        if source.get("llamafactory_sha256") != llama_sha256:
            raise MiningPoolError("candidate prescreen LLaMA-Factory hash mismatch")
        selection = prescreen.get("rollout_priority_selection", {})
        lines = selection.get("source_line_numbers")
        sample_ids = selection.get("sample_ids")
        if (
            selection.get("selection_label") != "AUTO_ACCEPT"
            or selection.get("semantically_grounded") is not False
            or not isinstance(lines, list)
            or not all(isinstance(value, int) and value > 0 for value in lines)
            or lines != sorted(lines)
            or len(lines) != len(set(lines))
            or not isinstance(sample_ids, list)
            or not all(isinstance(value, str) and value for value in sample_ids)
            or len(sample_ids) != len(set(sample_ids))
            or len(lines) != len(sample_ids)
        ):
            raise MiningPoolError("candidate prescreen AUTO_ACCEPT selection is invalid")
        if selection.get("source_line_numbers_sha256") != _sequence_sha256(lines):
            raise MiningPoolError("candidate prescreen line-number hash mismatch")
        if selection.get("sample_ids_sha256") != _sequence_sha256(sample_ids):
            raise MiningPoolError("candidate prescreen sample-ID hash mismatch")
        resolved_ids = [sample_ids_by_line.get(line) for line in lines]
        if resolved_ids != sample_ids:
            raise MiningPoolError("candidate prescreen IDs no longer align with train")
        available_lines = {line_number for line_number, _ in eligible}
        outside_remaining = set(lines) - available_lines
        if outside_remaining:
            raise MiningPoolError(
                "candidate prescreen overlaps an exclusion or source drifted at lines "
                f"{sorted(outside_remaining)[:5]}"
            )
        candidate_lines = set(lines)
        eligible = [item for item in eligible if item[0] in candidate_lines]
        candidate_prescreen = {
            "manifest_file": _display_path(candidate_prescreen_manifest_path),
            "manifest_sha256": _sha256_file(candidate_prescreen_manifest_path),
            "selection_label": "AUTO_ACCEPT",
            "semantically_grounded": False,
            "candidate_sample_count": len(lines),
            "source_line_numbers_sha256": selection[
                "source_line_numbers_sha256"
            ],
            "sample_ids_sha256": selection["sample_ids_sha256"],
            "purpose": "rollout prioritization only; final gate remains required",
        }

    if sample_count <= 0:
        raise MiningPoolError("sample_count must be positive")
    if sample_count > len(eligible):
        raise MiningPoolError(
            f"sample_count={sample_count} exceeds eligible_count={len(eligible)}"
        )

    sft_excluded_ids = [
        sample_ids_by_line[line_number]
        for line_number in sorted(sft_line_numbers)
    ]
    previous_exclusions: list[dict[str, Any]] = []
    for path, previous_manifest in previous_manifests:
        lines = previous_manifest["selected_source_line_numbers"]
        resolved_ids = [sample_ids_by_line[line_number] for line_number in lines]
        if resolved_ids != previous_manifest["sample_ids"]:
            raise MiningPoolError(
                f"{path}: sample IDs no longer align with canonical source lines"
            )
        if previous_manifest.get("source_sha256") != canonical_sha256:
            raise MiningPoolError(f"{path}: canonical source hash mismatch")
        if previous_manifest.get("llamafactory_source_sha256") != llama_sha256:
            raise MiningPoolError(f"{path}: LLaMA-Factory source hash mismatch")
        previous_exclusions.append(
            {
                "name": path.stem,
                "manifest_file": _display_path(path),
                "manifest_sha256": _sha256_file(path),
                "sample_count": len(resolved_ids),
                "selected_source_line_numbers_sha256": _sequence_sha256(lines),
                "sample_ids_sha256": _sequence_sha256(resolved_ids),
                "sample_ids": resolved_ids,
            }
        )

    selected_positions = sorted(
        random.Random(seed).sample(range(len(eligible)), sample_count)
    )
    selected = [eligible[position] for position in selected_positions]
    selected_lines = [line_number for line_number, _ in selected]
    selected_ids = [sample_id for _, sample_id in selected]
    if set(selected_ids) & set(excluded_ids):
        raise MiningPoolError("selected mining IDs overlap an excluded data source")
    output_sha256 = _write_selected_rows(
        canonical_train_path,
        output_path,
        set(selected_lines),
    )

    manifest: dict[str, Any] = {
        "manifest_version": "toolalign-dpo-mining-pool-v1",
        "purpose": "train-side SFT-v1 hard-negative mining",
        "source": "train",
        "source_file": _display_path(canonical_train_path),
        "source_sample_count": source_count,
        "source_sha256": canonical_sha256,
        "llamafactory_source_file": _display_path(llama_train_path),
        "llamafactory_source_sha256": llama_sha256,
        "canonical_llamafactory_full_alignment_verified": True,
        "exclusion": {
            "name": "sft_v1_10k",
            "manifest_file": _display_path(sft_manifest_path),
            "manifest_sha256": _sha256_file(sft_manifest_path),
            "sample_count": len(sft_excluded_ids),
            "sample_ids_sha256": _sequence_sha256(sft_excluded_ids),
            "sample_ids": sft_excluded_ids,
        },
        "previous_mining_exclusions": previous_exclusions,
        "excluded_sample_count_total": len(excluded_ids),
        "eligible_sample_count": len(eligible),
        "eligible_before_candidate_filter_count": eligible_before_candidate_filter,
        "candidate_prescreen": candidate_prescreen,
        "seed": seed,
        "sample_count": sample_count,
        "sampling_method": SAMPLING_METHOD,
        "sampling": "uniform_without_replacement",
        "output_order": "source_line_order",
        "selected_source_line_numbers_base": 1,
        "selected_source_line_numbers_sha256": _sequence_sha256(selected_lines),
        "selected_source_line_numbers": selected_lines,
        "sample_ids_sha256": _sequence_sha256(selected_ids),
        "sample_ids": selected_ids,
        "output_file": _display_path(output_path),
        "output_sha256": output_sha256,
        "test_data_used": False,
        "preference_pairs_generated": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 5k train-side pool for SFT-v1 hard-negative mining."
    )
    parser.add_argument("--canonical-train", type=Path, default=DEFAULT_CANONICAL_TRAIN)
    parser.add_argument("--llamafactory-train", type=Path, default=DEFAULT_LLAMA_TRAIN)
    parser.add_argument("--sft-manifest", type=Path, default=DEFAULT_SFT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude-mining-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Previously sampled mining manifest to exclude; repeat for multiple "
            "disjoint batches."
        ),
    )
    parser.add_argument(
        "--candidate-prescreen-manifest",
        type=Path,
        default=None,
        help=(
            "Optional Phase 5F rule-triage manifest; sampling is restricted to "
            "its AUTO_ACCEPT rollout-priority selection."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mining_pool(
        args.canonical_train.resolve(),
        args.llamafactory_train.resolve(),
        args.sft_manifest.resolve(),
        args.output.resolve(),
        args.manifest.resolve(),
        sample_count=args.sample_count,
        seed=args.seed,
        previous_mining_manifest_paths=[
            path.resolve() for path in args.exclude_mining_manifest
        ],
        candidate_prescreen_manifest_path=(
            args.candidate_prescreen_manifest.resolve()
            if args.candidate_prescreen_manifest is not None
            else None
        ),
    )
    print("DPO-v1 mining pool ready")
    print(f"source samples: {manifest['source_sample_count']}")
    print(f"excluded SFT-v1 samples: {manifest['exclusion']['sample_count']}")
    print(
        "excluded previous mining samples: "
        f"{sum(item['sample_count'] for item in manifest['previous_mining_exclusions'])}"
    )
    print(f"eligible samples: {manifest['eligible_sample_count']}")
    print(f"seed: {manifest['seed']}")
    print(f"selected samples: {manifest['sample_count']}")
    print(f"output: {args.output.resolve()}")
    print(f"manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()

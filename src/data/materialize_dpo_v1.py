"""Materialize the frozen DPO-v1 grounded preference inventory.

The output uses LLaMA-Factory's pairwise ShareGPT format.  Canonical gold
responses are encoded as ``function_call`` messages, while rejected responses
retain the exact text produced by SFT-v1 as an assistant message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.export_llamafactory import export_sample


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_POOL = PROJECT_ROOT / "outputs" / "dpo" / "dpo_v1_grounded_primary_pool.json"
NEW_CASES = (
    PROJECT_ROOT / "outputs" / "dpo" / "mining_v3_auto_accept" / "groundability_cases.jsonl"
)
SOURCE_POOLS = {
    "mining_v1": PROJECT_ROOT / "data" / "dpo" / "mining_pool_v1_5k.jsonl",
    "mining_v2": PROJECT_ROOT / "data" / "dpo" / "mining_pool_v2_5k.jsonl",
    "mining_v3_auto_accept": (
        PROJECT_ROOT / "data" / "dpo" / "mining_pool_v3_auto_accept_5k.jsonl"
    ),
}
EVALUATIONS = {
    "mining_v1": PROJECT_ROOT / "outputs" / "dpo" / "mining_v1" / "evaluation.jsonl",
    "mining_v2": PROJECT_ROOT / "outputs" / "dpo" / "mining_v2" / "evaluation.jsonl",
    "mining_v3_auto_accept": (
        PROJECT_ROOT / "outputs" / "dpo" / "mining_v3_auto_accept" / "evaluation.jsonl"
    ),
}
TRAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
SFT_MANIFEST = PROJECT_ROOT / "data" / "llamafactory" / "train_sft_v1_10k.manifest.json"
OUTPUT = PROJECT_ROOT / "data" / "llamafactory" / "dpo_v1_grounded_63.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "dpo" / "dpo_v1_grounded_63.manifest.json"


class MaterializationError(ValueError):
    """Raised when the frozen inventory fails an integrity assertion."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise MaterializationError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise MaterializationError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise MaterializationError(f"{path}:{line_number}: expected JSON object")
            yield value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _index_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise MaterializationError(f"{path}: record missing string sample_id")
        if sample_id in indexed:
            raise MaterializationError(f"{path}: duplicate sample_id {sample_id}")
        indexed[sample_id] = record
    return indexed


def _canonical_response(sample: dict[str, Any]) -> dict[str, str]:
    exported = export_sample(sample)
    response = exported["conversations"][-1]
    if response["from"] != "function_call":
        raise MaterializationError(f"{sample['sample_id']}: chosen response is not a tool call")
    return response


def _sft_selected_ids(train_path: Path, manifest_path: Path) -> set[str]:
    selected_lines = set(_read_json(manifest_path)["selected_line_numbers"])
    selected_ids: set[str] = set()
    for line_number, record in enumerate(_read_jsonl(train_path), start=1):
        if line_number in selected_lines:
            selected_ids.add(record["sample_id"])
    if len(selected_ids) != len(selected_lines):
        raise MaterializationError("SFT-v1 selected line numbers do not resolve uniquely")
    return selected_ids


def _load_inventory(old_pool_path: Path, new_cases_path: Path) -> list[dict[str, Any]]:
    old_pool = _read_json(old_pool_path)
    old_candidates = old_pool.get("grounded_candidates")
    if not isinstance(old_candidates, list) or len(old_candidates) != 53:
        raise MaterializationError("frozen grounded seed pool must contain exactly 53 candidates")

    inventory: list[dict[str, Any]] = []
    for candidate in old_candidates:
        if candidate.get("final_preference_decision") != "grounded_high_confidence":
            raise MaterializationError(f"{candidate.get('source_sample_id')}: invalid old decision")
        if candidate.get("dpo_v1_primary") is not True:
            raise MaterializationError(f"{candidate.get('source_sample_id')}: old candidate is not primary")
        if candidate.get("groundability", {}).get("passed") is not True:
            raise MaterializationError(f"{candidate.get('source_sample_id')}: old grounding did not pass")
        inventory.append({**candidate, "inventory_source": "grounded_seed_53"})

    new_candidates = [
        record
        for record in _read_jsonl(new_cases_path)
        if record.get("difference_groundability_triage", {}).get("decision") == "PASS"
    ]
    if len(new_candidates) != 10:
        raise MaterializationError("Phase 5F inventory must contain exactly 10 PASS candidates")
    for candidate in new_candidates:
        inventory.append(
            {
                **candidate,
                "source_batch": "mining_v3_auto_accept",
                "inventory_source": "phase_5f_verifier_pass_10",
            }
        )
    return inventory


def materialize(
    *,
    old_pool_path: Path = OLD_POOL,
    new_cases_path: Path = NEW_CASES,
    output_path: Path = OUTPUT,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    inventory = _load_inventory(old_pool_path, new_cases_path)
    sample_ids = [candidate["source_sample_id"] for candidate in inventory]
    if len(sample_ids) != 63 or len(set(sample_ids)) != 63:
        raise MaterializationError("combined inventory must contain 63 unique sample IDs")

    source_indexes = {name: _index_jsonl(path) for name, path in SOURCE_POOLS.items()}
    evaluation_indexes = {name: _index_jsonl(path) for name, path in EVALUATIONS.items()}
    train_ids = set(_index_jsonl(TRAIN))
    sft_ids = _sft_selected_ids(TRAIN, SFT_MANIFEST)

    output_records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    call_type_counts: Counter[str] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    for candidate in inventory:
        sample_id = candidate["source_sample_id"]
        source_batch = candidate["source_batch"]
        if source_batch not in source_indexes:
            raise MaterializationError(f"{sample_id}: unknown source batch {source_batch}")
        if sample_id not in train_ids:
            raise MaterializationError(f"{sample_id}: not present in canonical train")
        if sample_id in sft_ids:
            raise MaterializationError(f"{sample_id}: overlaps SFT-v1 10k")

        try:
            source = source_indexes[source_batch][sample_id]
            evaluation = evaluation_indexes[source_batch][sample_id]
        except KeyError as error:
            raise MaterializationError(f"{sample_id}: missing source/evaluation record") from error

        canonical_gold = candidate["canonical_gold"]
        raw_rejected = candidate["sft_raw_prediction"]
        if source.get("assistant") != canonical_gold or evaluation.get("gold") != canonical_gold:
            raise MaterializationError(f"{sample_id}: chosen does not equal canonical gold")
        if evaluation.get("raw_prediction") != raw_rejected:
            raise MaterializationError(f"{sample_id}: rejected does not equal frozen SFT prediction")
        metrics = evaluation.get("metrics", {})
        if metrics.get("full_call_exact") is not False or "argument_mismatch" not in evaluation.get("errors", []):
            raise MaterializationError(f"{sample_id}: rejected is not a frozen argument failure")

        exported = export_sample(source)
        prompt = exported["conversations"][:-1]
        chosen = _canonical_response(source)
        rejected = {"from": "gpt", "value": raw_rejected}
        if chosen["value"] == rejected["value"]:
            raise MaterializationError(f"{sample_id}: chosen and rejected are identical")

        diagnostics = sorted(set(candidate.get("argument_diagnostics", [])))
        output_records.append(
            {
                "conversations": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "system": exported["system"],
                "tools": exported["tools"],
                "metadata": {
                    "source_sample_id": sample_id,
                    "source_split": "train",
                    "source_batch": source_batch,
                    "generator": "sft_v1_deterministic",
                    "failure_type": "argument_mismatch",
                    "argument_diagnostics": diagnostics,
                    "semantic_bucket": candidate.get("semantic_bucket"),
                    "preference_quality": "grounded_high_confidence",
                    "inventory_source": candidate["inventory_source"],
                },
            }
        )
        source_counts[candidate["inventory_source"]] += 1
        call_type_counts[candidate["call_type"]] += 1
        diagnostic_counts.update(diagnostics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in output_records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    input_paths = [old_pool_path, new_cases_path, *SOURCE_POOLS.values(), *EVALUATIONS.values(), TRAIN, SFT_MANIFEST]
    manifest = {
        "manifest_version": "toolalign-dpo-v1-grounded-63-v1",
        "status": "materialized_small_scale_dpo_experiment",
        "description": "Small, high-confidence, grounded, on-policy residual preference set.",
        "sample_count": len(output_records),
        "ordered_sample_ids_sha256": _sha256_ids(sample_ids),
        "sample_ids": sample_ids,
        "inventory_counts": dict(sorted(source_counts.items())),
        "call_type_counts": dict(sorted(call_type_counts.items())),
        "diagnostic_occurrences": dict(sorted(diagnostic_counts.items())),
        "serialization": {
            "format": "llamafactory_sharegpt_pairwise_jsonl",
            "chosen": "canonical gold encoded as function_call",
            "rejected": "exact SFT-v1 raw generation encoded as gpt content",
            "template": "qwen3_nothink",
        },
        "integrity_checks": {
            "all_source_samples_in_train": True,
            "no_overlap_with_sft_v1_10k": True,
            "chosen_equals_canonical_gold": True,
            "rejected_equals_frozen_sft_prediction": True,
            "chosen_differs_from_rejected": True,
            "all_rejected_fail_frozen_evaluator": True,
            "all_failures_include_argument_mismatch": True,
            "all_groundability_decisions_pass": True,
            "sample_ids_unique": True,
        },
        "provenance_note": (
            "The 53-sample seed passed the corrected difference-level groundability gate; "
            "the additional 10 samples are verifier-backed Phase 5F PASS cases. This artifact "
            "must not be described as 63 fully human-annotated samples."
        ),
        "inputs_sha256": {_display_path(path): _sha256_file(path) for path in input_paths},
        "output": _display_path(output_path),
        "output_sha256": _sha256_file(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = materialize(output_path=args.output, manifest_path=args.manifest)
    print(json.dumps({"sample_count": manifest["sample_count"], "output": manifest["output"]}, indent=2))


if __name__ == "__main__":
    main()

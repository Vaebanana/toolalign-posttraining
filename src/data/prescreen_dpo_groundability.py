"""Apply cheap rule-based groundability risk triage to unused train samples.

Phase 5F intentionally stops after writing a deterministic rule-triage
manifest.  It does not run inference, generate predictions, materialize
preference pairs, or create trainer configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from src.data.build_dpo_mining_pool import (
        _display_path,
        _inspect_aligned_sources,
        _load_json_object,
        _load_previous_mining_manifests,
        _load_sft_manifest,
        _sequence_sha256,
        _sha256_file,
    )
    from src.eval.dpo_groundability import (
        GROUNDABILITY_VERSION,
        triage_gold_sample_groundability,
    )
except ModuleNotFoundError:  # Support direct script execution.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.build_dpo_mining_pool import (
        _display_path,
        _inspect_aligned_sources,
        _load_json_object,
        _load_previous_mining_manifests,
        _load_sft_manifest,
        _sequence_sha256,
        _sha256_file,
    )
    from src.eval.dpo_groundability import (
        GROUNDABILITY_VERSION,
        triage_gold_sample_groundability,
    )


PRESCREEN_VERSION = "toolalign-dpo-groundability-rule-triage-v1"
DEFAULT_CANONICAL_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_LLAMA_TRAIN = PROJECT_ROOT / "data" / "llamafactory" / "train.jsonl"
DEFAULT_SFT_MANIFEST = (
    PROJECT_ROOT / "data" / "llamafactory" / "train_sft_v1_10k.manifest.json"
)
DEFAULT_MINING_MANIFESTS = (
    PROJECT_ROOT / "data" / "dpo" / "mining_pool_v1_5k.manifest.json",
    PROJECT_ROOT / "data" / "dpo" / "mining_pool_v2_5k.manifest.json",
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "dpo" / "groundability_prescreen_v1.manifest.json"
)


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _distribution(
    remaining: Counter[tuple[str, str]],
    decisions: dict[str, Counter[tuple[str, str]]],
    *,
    primary_index: int,
    secondary_index: int,
) -> dict[str, Any]:
    primary_values = sorted({key[primary_index] for key in remaining})
    result: dict[str, Any] = {}
    for primary in primary_values:
        remaining_count = sum(
            count for key, count in remaining.items() if key[primary_index] == primary
        )
        decision_counts = {
            decision: sum(
                count
                for key, count in distribution.items()
                if key[primary_index] == primary
            )
            for decision, distribution in decisions.items()
        }
        secondary_values = sorted(
            {
                key[secondary_index]
                for key in remaining
                if key[primary_index] == primary
            }
        )
        result[primary] = {
            "remaining_samples": remaining_count,
            "decision_counts": decision_counts,
            "auto_accept_rate": _rate(
                decision_counts["AUTO_ACCEPT"], remaining_count
            ),
            "breakdown": {
                secondary: {
                    "remaining_samples": remaining[(primary, secondary)]
                    if primary_index == 0
                    else remaining[(secondary, primary)],
                    "decision_counts": {
                        decision: distribution[(primary, secondary)]
                        if primary_index == 0
                        else distribution[(secondary, primary)]
                        for decision, distribution in decisions.items()
                    },
                }
                for secondary in secondary_values
            },
        }
    return result


def _validate_exclusion_manifests(
    *,
    canonical_train_path: Path,
    llama_train_path: Path,
    sft_manifest_path: Path,
    mining_manifest_paths: list[Path],
) -> tuple[
    dict[str, Any],
    list[tuple[Path, dict[str, Any]]],
    set[int],
    dict[int, str],
    str,
    str,
]:
    sft_manifest = _load_sft_manifest(sft_manifest_path, llama_train_path)
    mining_manifests = _load_previous_mining_manifests(mining_manifest_paths)
    sft_lines = set(sft_manifest["selected_line_numbers"])
    mining_lines: set[int] = set()
    for path, manifest in mining_manifests:
        current = set(manifest["selected_source_line_numbers"])
        overlap = current & (sft_lines | mining_lines)
        if overlap:
            raise ValueError(
                f"{path}: overlaps an earlier exclusion at lines {sorted(overlap)[:5]}"
            )
        mining_lines.update(current)
    excluded_lines = sft_lines | mining_lines
    (
        canonical_sha256,
        llama_sha256,
        remaining,
        _,
        sample_ids_by_line,
    ) = _inspect_aligned_sources(
        canonical_train_path,
        llama_train_path,
        excluded_lines,
    )
    if len(remaining) + len(excluded_lines) != sft_manifest["source_sample_count"]:
        raise ValueError("remaining and excluded samples do not close to train")
    for path, manifest in mining_manifests:
        lines = manifest["selected_source_line_numbers"]
        if manifest.get("source_sha256") != canonical_sha256:
            raise ValueError(f"{path}: canonical source hash mismatch")
        if manifest.get("llamafactory_source_sha256") != llama_sha256:
            raise ValueError(f"{path}: LLaMA-Factory source hash mismatch")
        if [sample_ids_by_line[line] for line in lines] != manifest["sample_ids"]:
            raise ValueError(f"{path}: sample IDs no longer match source lines")
    return (
        sft_manifest,
        mining_manifests,
        excluded_lines,
        sample_ids_by_line,
        canonical_sha256,
        llama_sha256,
    )


def build_groundability_rule_triage(
    *,
    canonical_train_path: Path,
    llama_train_path: Path,
    sft_manifest_path: Path,
    mining_manifest_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Rule-triage every unused train sample without semantic adjudication."""
    for path in (
        canonical_train_path,
        llama_train_path,
        sft_manifest_path,
        *mining_manifest_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    if output_path.resolve() in {
        canonical_train_path.resolve(),
        llama_train_path.resolve(),
    }:
        raise ValueError("output path must not overwrite a source")

    (
        sft_manifest,
        mining_manifests,
        excluded_lines,
        sample_ids_by_line,
        canonical_sha256,
        llama_sha256,
    ) = _validate_exclusion_manifests(
        canonical_train_path=canonical_train_path,
        llama_train_path=llama_train_path,
        sft_manifest_path=sft_manifest_path,
        mining_manifest_paths=mining_manifest_paths,
    )

    remaining_distribution: Counter[tuple[str, str]] = Counter()
    decision_distributions = {
        decision: Counter()
        for decision in ("AUTO_ACCEPT", "AUTO_REJECT", "SEMANTIC_UNKNOWN")
    }
    decision_counts = Counter()
    reason_counts = Counter()
    reasons_by_source: dict[str, Counter[str]] = {}
    reasons_by_call_type: dict[str, Counter[str]] = {}
    auto_accept_evidence_counts = Counter()
    auto_accept_lines: list[int] = []
    auto_accept_ids: list[str] = []
    remaining_ids: list[str] = []

    with canonical_train_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if line_number in excluded_lines:
                continue
            sample = _load_json_object(raw, canonical_train_path, line_number)
            sample_id = sample.get("sample_id")
            if sample_id != sample_ids_by_line[line_number]:
                raise ValueError(f"{canonical_train_path}:{line_number}: ID drift")
            source = sample.get("source")
            if not isinstance(source, str) or not source:
                raise TypeError(
                    f"{canonical_train_path}:{line_number}: invalid source"
                )
            assessment = triage_gold_sample_groundability(sample)
            call_type = assessment["call_type"]
            remaining_distribution[(source, call_type)] += 1
            remaining_ids.append(sample_id)
            decision = assessment["decision"]
            reason = assessment["reason"]
            decision_counts[decision] += 1
            decision_distributions[decision][(source, call_type)] += 1
            reason_counts[reason] += 1
            reasons_by_source.setdefault(source, Counter())[reason] += 1
            reasons_by_call_type.setdefault(call_type, Counter())[reason] += 1
            if decision == "AUTO_ACCEPT":
                auto_accept_evidence_counts.update(assessment["evidence_counts"])
                auto_accept_lines.append(line_number)
                auto_accept_ids.append(sample_id)

    source_count = sft_manifest["source_sample_count"]
    remaining_count = len(remaining_ids)
    auto_accept_count = len(auto_accept_ids)
    expected_remaining = source_count - len(excluded_lines)
    if remaining_count != expected_remaining:
        raise ValueError(
            f"expected {expected_remaining} remaining rows, found {remaining_count}"
        )
    if sum(decision_counts.values()) != remaining_count:
        raise ValueError("triage decisions do not close to remaining train")
    if auto_accept_count != decision_counts["AUTO_ACCEPT"]:
        raise ValueError("AUTO_ACCEPT selection count differs from triage count")
    if auto_accept_lines != sorted(auto_accept_lines):
        raise ValueError("AUTO_ACCEPT line numbers are not source ordered")
    if len(auto_accept_ids) != len(set(auto_accept_ids)):
        raise ValueError("AUTO_ACCEPT sample IDs are not unique")

    mining_exclusions = [
        {
            "manifest_file": _display_path(path),
            "manifest_sha256": _sha256_file(path),
            "sample_count": manifest["sample_count"],
            "sample_ids_sha256": manifest["sample_ids_sha256"],
            "selected_source_line_numbers_sha256": manifest[
                "selected_source_line_numbers_sha256"
            ],
        }
        for path, manifest in mining_manifests
    ]
    output: dict[str, Any] = {
        "manifest_version": PRESCREEN_VERSION,
        "phase": "phase_5f_high_precision_rule_triage",
        "status": "rule_triage_complete_no_rollout",
        "purpose": "reduce obvious grounding risk and prioritize rollout candidates",
        "policy": {
            "scope": "prompt_and_canonical_gold_before_prediction",
            "three_way_decisions": [
                "AUTO_ACCEPT",
                "AUTO_REJECT",
                "SEMANTIC_UNKNOWN",
            ],
            "auto_accept_definition": (
                "Every canonical gold argument leaf matched a narrow "
                "verifier-backed rule. This is rollout prioritization, not a "
                "semantic-groundability verdict."
            ),
            "auto_reject_definition": (
                "A non-tool target or an obvious undocumented opaque/external "
                "mapping was detected."
            ),
            "semantic_unknown_definition": (
                "Rules abstain; no semantic preference direction is asserted."
            ),
            "auto_accept_evidence": [
                "query literal",
                "deterministic query plus schema transformation",
                "explicit schema option or mapping",
                "schema-declared standard mapping",
            ],
            "auto_reject_risk_signals": [
                "external entity-to-ID, UUID, URI, ticker, or private-code mapping",
                "non-tool-call target",
            ],
            "not_claimed": [
                "semantic groundability",
                "chosen is uniquely recoverable",
                "preference direction",
                "groundability gate accuracy",
            ],
            "rule_triage_may_have_false_positives": True,
            "rule_triage_may_have_false_negatives": True,
            "final_difference_level_groundability_gate_still_required": True,
            "difference_level_groundability_version": GROUNDABILITY_VERSION,
        },
        "source": {
            "name": "train",
            "canonical_file": _display_path(canonical_train_path),
            "canonical_sha256": canonical_sha256,
            "llamafactory_file": _display_path(llama_train_path),
            "llamafactory_sha256": llama_sha256,
            "canonical_llamafactory_full_alignment_verified": True,
            "sample_count": source_count,
        },
        "exclusions": {
            "sft_v1": {
                "manifest_file": _display_path(sft_manifest_path),
                "manifest_sha256": _sha256_file(sft_manifest_path),
                "sample_count": sft_manifest["sample_count"],
                "selected_line_numbers_sha256": sft_manifest[
                    "selected_line_numbers_sha256"
                ],
            },
            "previous_deterministic_mining": mining_exclusions,
            "excluded_sample_count_total": len(excluded_lines),
            "all_exclusion_sets_disjoint": True,
        },
        "inventory": {
            "remaining_train_samples": remaining_count,
            "decision_counts": {
                decision: decision_counts[decision]
                for decision in ("AUTO_ACCEPT", "AUTO_REJECT", "SEMANTIC_UNKNOWN")
            },
            "auto_accept_rate": _rate(auto_accept_count, remaining_count),
            "closure_verified": sum(decision_counts.values()) == remaining_count,
            "by_source": _distribution(
                remaining_distribution,
                decision_distributions,
                primary_index=0,
                secondary_index=1,
            ),
            "by_call_type": _distribution(
                remaining_distribution,
                decision_distributions,
                primary_index=1,
                secondary_index=0,
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
            "reasons_by_source": {
                key: dict(sorted(value.items()))
                for key, value in sorted(reasons_by_source.items())
            },
            "reasons_by_call_type": {
                key: dict(sorted(value.items()))
                for key, value in sorted(reasons_by_call_type.items())
            },
            "auto_accept_evidence_occurrences": dict(
                sorted(auto_accept_evidence_counts.items())
            ),
        },
        "rollout_priority_selection": {
            "selection_label": "AUTO_ACCEPT",
            "semantically_grounded": False,
            "output_order": "canonical_source_line_order",
            "source_line_numbers_base": 1,
            "source_line_numbers_sha256": _sequence_sha256(auto_accept_lines),
            "sample_ids_sha256": _sequence_sha256(auto_accept_ids),
            "source_line_numbers": auto_accept_lines,
            "sample_ids": auto_accept_ids,
        },
        "operations": {
            "rollout_started": False,
            "predictions_generated": False,
            "frozen_evaluation_run": False,
            "preference_pairs_generated": False,
            "dpo_train_written": False,
            "trainer_created": False,
        },
        "test_data_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-train", type=Path, default=DEFAULT_CANONICAL_TRAIN)
    parser.add_argument("--llamafactory-train", type=Path, default=DEFAULT_LLAMA_TRAIN)
    parser.add_argument("--sft-manifest", type=Path, default=DEFAULT_SFT_MANIFEST)
    parser.add_argument(
        "--mining-manifest",
        type=Path,
        action="append",
        dest="mining_manifests",
        help="Repeat for each prior deterministic mining batch.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_groundability_rule_triage(
        canonical_train_path=args.canonical_train.resolve(),
        llama_train_path=args.llamafactory_train.resolve(),
        sft_manifest_path=args.sft_manifest.resolve(),
        mining_manifest_paths=[
            path.resolve()
            for path in (args.mining_manifests or DEFAULT_MINING_MANIFESTS)
        ],
        output_path=args.output.resolve(),
    )
    inventory = result["inventory"]
    print("Phase 5F high-precision rule triage complete")
    print(f"remaining={inventory['remaining_train_samples']}")
    for decision, count in inventory["decision_counts"].items():
        print(f"{decision}={count}")
    print(f"output={args.output.resolve()}")
    print("rollout_started=false")


if __name__ == "__main__":
    main()

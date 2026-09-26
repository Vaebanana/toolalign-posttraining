"""Materialize the audited Phase 5C wrong-value adjudication results.

This phase reviews the frozen Phase 5B queue.  It does not construct chosen /
rejected pairs, write a DPO training dataset, or create training configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.eval.analyze_dpo_mining import PRIMARY_BUCKETS
    from src.eval.analyze_errors import CALL_TYPES, SEMANTIC_BUCKETS
except ModuleNotFoundError:
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))
    from src.eval.analyze_dpo_mining import PRIMARY_BUCKETS  # type: ignore[no-redef]
    from src.eval.analyze_errors import (  # type: ignore[no-redef]
        CALL_TYPES,
        SEMANTIC_BUCKETS,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINING_ROOT = PROJECT_ROOT / "outputs" / "dpo" / "mining_v1"
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "configs"
    / "analysis"
    / "dpo_mining_v1_wrong_value_adjudications.json"
)
DECISIONS = ("high_confidence", "excluded", "uncertain")
EXPECTED_PROVENANCE = "codex_assisted_case_by_case_review_not_human_ground_truth"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _ordered_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _phase_5c_queue(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = [
        case
        for case in cases
        if case.get("preference_filter", {}).get("decision")
        == "requires_semantic_review"
        and "wrong_value" in case.get("argument_diagnostics", [])
    ]
    sample_ids = [case.get("source_sample_id") for case in queue]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
        raise TypeError("Phase 5C queue contains an invalid source_sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Phase 5C queue contains duplicate source_sample_id values")
    return queue


def _load_and_validate_annotations(
    path: Path,
    queue_ids: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: annotation document must be an object")
    provenance = value.get("annotation_provenance")
    if provenance != EXPECTED_PROVENANCE:
        raise ValueError(f"{path}: unexpected annotation_provenance")
    if value.get("queue_sample_count") != len(queue_ids):
        raise ValueError(f"{path}: declared queue count does not match Phase 5C queue")
    queue_hash = _ordered_ids_sha256(queue_ids)
    if value.get("queue_sample_ids_sha256") != queue_hash:
        raise ValueError(f"{path}: declared queue hash does not match Phase 5C queue")

    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError(f"{path}: annotations must be an array")
    annotation_ids: list[str] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise TypeError(f"{path}: annotation {index} must be an object")
        sample_id = annotation.get("sample_id")
        decision = annotation.get("decision")
        bucket = annotation.get("semantic_bucket")
        reason = annotation.get("preference_reason")
        primary = annotation.get("dpo_v1_primary")
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError(f"{path}: annotation {index} has invalid sample_id")
        if decision not in DECISIONS:
            raise ValueError(f"{path}: {sample_id} has invalid decision")
        if bucket not in SEMANTIC_BUCKETS:
            raise ValueError(f"{path}: {sample_id} has invalid semantic_bucket")
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError(f"{path}: {sample_id} has invalid preference_reason")
        if not isinstance(primary, bool):
            raise TypeError(f"{path}: {sample_id} has invalid dpo_v1_primary")
        expected_primary = decision == "high_confidence" and bucket in PRIMARY_BUCKETS
        if primary is not expected_primary:
            raise ValueError(
                f"{path}: {sample_id} dpo_v1_primary must equal high-confidence "
                "membership in a frozen primary bucket"
            )
        annotation_ids.append(sample_id)

    if annotation_ids != queue_ids:
        missing = sorted(set(queue_ids) - set(annotation_ids))
        extra = sorted(set(annotation_ids) - set(queue_ids))
        raise ValueError(
            f"{path}: annotations must cover the ordered Phase 5C queue exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return provenance, annotations


def adjudicate_wrong_values(
    *,
    candidate_cases_path: Path,
    filtering_summary_path: Path,
    annotations_path: Path,
    output_cases_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    cases = _read_jsonl(candidate_cases_path)
    queue = _phase_5c_queue(cases)
    queue_ids = [case["source_sample_id"] for case in queue]
    provenance, annotations = _load_and_validate_annotations(
        annotations_path, queue_ids
    )
    phase_5b = json.loads(filtering_summary_path.read_text(encoding="utf-8"))
    baseline_high = phase_5b.get("candidate_decision_counts", {}).get(
        "high_confidence"
    )
    baseline_primary = phase_5b.get("high_confidence_candidates", {}).get(
        "dpo_v1_primary_target_samples"
    )
    baseline_pending = phase_5b.get("candidate_decision_counts", {}).get(
        "requires_semantic_review"
    )
    structural_candidates = phase_5b.get("structural_candidate_samples")
    if not all(
        isinstance(value, int)
        for value in (
            baseline_high,
            baseline_primary,
            baseline_pending,
            structural_candidates,
        )
    ):
        raise TypeError(f"{filtering_summary_path}: invalid Phase 5B counts")
    if baseline_pending < len(queue):
        raise ValueError("Phase 5C queue exceeds the Phase 5B pending population")

    annotations_by_id = {
        annotation["sample_id"]: annotation for annotation in annotations
    }
    adjudicated_cases: list[dict[str, Any]] = []
    for case in queue:
        annotation = annotations_by_id[case["source_sample_id"]]
        adjudicated_cases.append(
            {
                **case,
                "record_type": "phase_5c_wrong_value_adjudication",
                "phase_5c_adjudication": {
                    "decision": annotation["decision"],
                    "semantic_bucket": annotation["semantic_bucket"],
                    "preference_reason": annotation["preference_reason"],
                    "review_note": annotation["preference_reason"],
                    "dpo_v1_primary": annotation["dpo_v1_primary"],
                    "annotation_provenance": provenance,
                    "candidate_for_human_verification": (
                        annotation["decision"] == "high_confidence"
                    ),
                },
                "preference_pair_materialized": False,
            }
        )

    decisions = Counter(
        case["phase_5c_adjudication"]["decision"] for case in adjudicated_cases
    )
    buckets = Counter(
        case["phase_5c_adjudication"]["semantic_bucket"]
        for case in adjudicated_cases
    )
    high_buckets = Counter(
        case["phase_5c_adjudication"]["semantic_bucket"]
        for case in adjudicated_cases
        if case["phase_5c_adjudication"]["decision"] == "high_confidence"
    )
    call_types = {
        decision: Counter(
            case["call_type"]
            for case in adjudicated_cases
            if case["phase_5c_adjudication"]["decision"] == decision
        )
        for decision in DECISIONS
    }
    phase_5c_high = decisions["high_confidence"]
    phase_5c_primary = sum(
        case["phase_5c_adjudication"]["dpo_v1_primary"]
        for case in adjudicated_cases
    )
    combined_high = baseline_high + phase_5c_high
    combined_primary = baseline_primary + phase_5c_primary
    remaining_unreviewed = baseline_pending - len(queue)

    summary = {
        "analysis": "dpo_v1_train_side_mining",
        "phase": "phase_5c_semantic_preference_adjudication",
        "scope": "phase_5b_pending_wrong_value_only",
        "queue": {
            "definition": (
                "preference_filter.decision == requires_semantic_review AND "
                "argument_diagnostics contains wrong_value"
            ),
            "samples": len(queue),
            "reviewed_samples": len(adjudicated_cases),
            "coverage": _rate(len(adjudicated_cases), len(queue)),
            "ordered_sample_ids_sha256": _ordered_ids_sha256(queue_ids),
        },
        "annotation": {
            "source": str(annotations_path.resolve()),
            "provenance": provenance,
            "human_ground_truth": False,
            "high_confidence_requires_human_quick_review_before_training": True,
        },
        "adjudication": {
            "decision_counts": {
                decision: decisions[decision] for decision in DECISIONS
            },
            "decision_rates": {
                decision: _rate(decisions[decision], len(adjudicated_cases))
                for decision in DECISIONS
            },
            "by_call_type": {
                decision: {
                    call_type: call_types[decision][call_type]
                    for call_type in CALL_TYPES
                }
                for decision in DECISIONS
            },
            "all_bucket_counts": {
                bucket: buckets[bucket] for bucket in SEMANTIC_BUCKETS
            },
            "high_confidence_bucket_counts": {
                bucket: high_buckets[bucket] for bucket in SEMANTIC_BUCKETS
            },
            "high_confidence_dpo_v1_primary_samples": phase_5c_primary,
            "high_confidence_secondary_samples": phase_5c_high - phase_5c_primary,
        },
        "updated_phase_5_population": {
            "structural_candidate_samples": structural_candidates,
            "phase_5b_existing_high_confidence": baseline_high,
            "phase_5c_new_high_confidence": phase_5c_high,
            "combined_high_confidence": combined_high,
            "phase_5b_existing_primary": baseline_primary,
            "phase_5c_new_primary": phase_5c_primary,
            "combined_dpo_v1_primary": combined_primary,
            "excluded_after_phase_5c": (
                phase_5b["candidate_decision_counts"]["excluded"]
                + decisions["excluded"]
            ),
            "phase_5c_uncertain": decisions["uncertain"],
            "phase_5b_pending_not_in_wrong_value_queue": remaining_unreviewed,
        },
        "stopping_condition": {
            "wrong_value_review_complete": len(adjudicated_cases) == len(queue),
            "combined_high_confidence_samples": combined_high,
            "threshold_for_direct_dpo_v1_experiment": 300,
            "threshold_met": combined_high >= 300,
            "expand_to_10k_now": False,
            "next_action": (
                "Review remaining Phase 5B pending missing_argument and "
                "extra_argument candidates; do not expand to 10k yet."
            ),
            "rationale": (
                "The 5k pool expansion decision is deferred until the remaining "
                "structural candidates have been adjudicated and total "
                "high-confidence yield is known."
            ),
        },
        "outputs": {
            "adjudicated_cases": str(output_cases_path.resolve()),
            "preference_pairs_generated": False,
            "dpo_train_written": False,
            "trainer_created": False,
        },
    }

    output_cases_path.parent.mkdir(parents=True, exist_ok=True)
    with output_cases_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in adjudicated_cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    _write_json(output_summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining-root", type=Path, default=DEFAULT_MINING_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mining_root = args.mining_root.resolve()
    summary = adjudicate_wrong_values(
        candidate_cases_path=mining_root / "candidate_cases.jsonl",
        filtering_summary_path=mining_root / "filtering_summary.json",
        annotations_path=args.annotations.resolve(),
        output_cases_path=mining_root / "wrong_value_adjudication.jsonl",
        output_summary_path=mining_root / "wrong_value_adjudication_summary.json",
    )
    decisions = summary["adjudication"]["decision_counts"]
    updated = summary["updated_phase_5_population"]
    print("Phase 5C semantic preference adjudication complete")
    print(
        f"reviewed={summary['queue']['reviewed_samples']}, "
        f"high_confidence={decisions['high_confidence']}, "
        f"excluded={decisions['excluded']}, uncertain={decisions['uncertain']}"
    )
    print(
        f"combined_high_confidence={updated['combined_high_confidence']}, "
        f"combined_dpo_v1_primary={updated['combined_dpo_v1_primary']}"
    )
    print(f"cases: {mining_root / 'wrong_value_adjudication.jsonl'}")
    print(f"summary: {mining_root / 'wrong_value_adjudication_summary.json'}")


if __name__ == "__main__":
    main()

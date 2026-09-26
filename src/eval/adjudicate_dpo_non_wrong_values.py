"""Materialize Phase 5D non-wrong-value preference adjudications.

The input queue is the remaining Phase 5B semantic-review population after
Phase 5C: pending structural candidates without a wrong_value diagnostic.
This script never materializes DPO pairs or creates trainer configuration.
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
    from src.eval.analyze_errors import CALL_TYPES
except ModuleNotFoundError:
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))
    from src.eval.analyze_errors import CALL_TYPES  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINING_ROOT = PROJECT_ROOT / "outputs" / "dpo" / "mining_v1"
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "configs"
    / "analysis"
    / "dpo_mining_v1_non_wrong_value_adjudications.json"
)
DECISIONS = ("high_confidence", "excluded", "uncertain")
DIAGNOSTIC_PRIORITY = (
    "missing_argument",
    "extra_argument",
    "nested_structure_mismatch",
    "type_mismatch",
)
EXPECTED_PROVENANCE = "codex_assisted_case_by_case_review_not_human_ground_truth"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


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


def _diagnostic_type(case: dict[str, Any]) -> str:
    diagnostics = case.get("argument_diagnostics")
    if not isinstance(diagnostics, list):
        raise TypeError(
            f"{case.get('source_sample_id')}: invalid argument_diagnostics"
        )
    try:
        return next(
            diagnostic
            for diagnostic in DIAGNOSTIC_PRIORITY
            if diagnostic in diagnostics
        )
    except StopIteration as error:
        raise ValueError(
            f"{case.get('source_sample_id')}: no Phase 5D diagnostic"
        ) from error


def _phase_5d_queue(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = [
        case
        for case in cases
        if case.get("preference_filter", {}).get("decision")
        == "requires_semantic_review"
        and "wrong_value" not in case.get("argument_diagnostics", [])
    ]
    sample_ids = [case.get("source_sample_id") for case in queue]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
        raise TypeError("Phase 5D queue contains an invalid source_sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Phase 5D queue contains duplicate source_sample_id values")
    for case in queue:
        _diagnostic_type(case)
    return queue


def _load_and_validate_annotations(
    path: Path,
    queue: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    value = _read_json(path)
    provenance = value.get("annotation_provenance")
    if provenance != EXPECTED_PROVENANCE:
        raise ValueError(f"{path}: unexpected annotation_provenance")
    if value.get("review_order") != list(DIAGNOSTIC_PRIORITY):
        raise ValueError(f"{path}: review_order does not match frozen priority")
    queue_ids = [case["source_sample_id"] for case in queue]
    if value.get("queue_sample_count") != len(queue):
        raise ValueError(f"{path}: declared queue count does not match Phase 5D queue")
    if value.get("queue_sample_ids_sha256") != _ordered_ids_sha256(queue_ids):
        raise ValueError(f"{path}: declared queue hash does not match Phase 5D queue")

    expected_review_queue = sorted(
        enumerate(queue),
        key=lambda indexed: (
            DIAGNOSTIC_PRIORITY.index(_diagnostic_type(indexed[1])),
            indexed[0],
        ),
    )
    expected_review_ids = [case["source_sample_id"] for _, case in expected_review_queue]
    case_by_id = {case["source_sample_id"]: case for case in queue}
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError(f"{path}: annotations must be an array")
    annotation_ids: list[str] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise TypeError(f"{path}: annotation {index} must be an object")
        sample_id = annotation.get("sample_id")
        decision = annotation.get("decision")
        diagnostic = annotation.get("diagnostic_type")
        semantic_reason = annotation.get("semantic_reason")
        primary = annotation.get("dpo_v1_primary")
        review_note = annotation.get("review_note")
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError(f"{path}: annotation {index} has invalid sample_id")
        if decision not in DECISIONS:
            raise ValueError(f"{path}: {sample_id} has invalid decision")
        if diagnostic not in DIAGNOSTIC_PRIORITY:
            raise ValueError(f"{path}: {sample_id} has invalid diagnostic_type")
        if not isinstance(semantic_reason, str) or not semantic_reason.strip():
            raise TypeError(f"{path}: {sample_id} has invalid semantic_reason")
        if not isinstance(review_note, str) or not review_note.strip():
            raise TypeError(f"{path}: {sample_id} has invalid review_note")
        if not isinstance(primary, bool):
            raise TypeError(f"{path}: {sample_id} has invalid dpo_v1_primary")
        if primary and decision != "high_confidence":
            raise ValueError(
                f"{path}: {sample_id} can be primary only when high_confidence"
            )
        if sample_id in case_by_id and diagnostic != _diagnostic_type(
            case_by_id[sample_id]
        ):
            raise ValueError(
                f"{path}: {sample_id} diagnostic_type does not match frozen priority"
            )
        annotation_ids.append(sample_id)

    if annotation_ids != expected_review_ids:
        missing = sorted(set(queue_ids) - set(annotation_ids))
        extra = sorted(set(annotation_ids) - set(queue_ids))
        raise ValueError(
            f"{path}: annotations must cover the priority-ordered Phase 5D queue "
            f"exactly; missing={missing[:5]}, extra={extra[:5]}"
        )
    return provenance, annotations


def adjudicate_non_wrong_values(
    *,
    candidate_cases_path: Path,
    candidate_summary_path: Path,
    filtering_summary_path: Path,
    phase_5c_summary_path: Path,
    annotations_path: Path,
    output_cases_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    candidate_cases = _read_jsonl(candidate_cases_path)
    queue = _phase_5d_queue(candidate_cases)
    queue_ids = [case["source_sample_id"] for case in queue]
    provenance, annotations = _load_and_validate_annotations(
        annotations_path, queue
    )
    candidate_summary = _read_json(candidate_summary_path)
    phase_5b = _read_json(filtering_summary_path)
    phase_5c = _read_json(phase_5c_summary_path)
    cases_by_id = {case["source_sample_id"]: case for case in queue}

    reviewed_cases: list[dict[str, Any]] = []
    for annotation in annotations:
        case = cases_by_id[annotation["sample_id"]]
        reviewed_cases.append(
            {
                **case,
                "record_type": "phase_5d_non_wrong_value_adjudication",
                "phase_5d_adjudication": {
                    "decision": annotation["decision"],
                    "diagnostic_type": annotation["diagnostic_type"],
                    "semantic_reason": annotation["semantic_reason"],
                    "dpo_v1_primary": annotation["dpo_v1_primary"],
                    "review_note": annotation["review_note"],
                    "annotation_provenance": provenance,
                    "candidate_for_human_verification": (
                        annotation["decision"] == "high_confidence"
                    ),
                },
                "preference_pair_materialized": False,
            }
        )

    decisions = Counter(
        case["phase_5d_adjudication"]["decision"] for case in reviewed_cases
    )
    diagnostics = Counter(
        case["phase_5d_adjudication"]["diagnostic_type"]
        for case in reviewed_cases
    )
    high_diagnostics = Counter(
        case["phase_5d_adjudication"]["diagnostic_type"]
        for case in reviewed_cases
        if case["phase_5d_adjudication"]["decision"] == "high_confidence"
    )
    primary_diagnostics = Counter(
        case["phase_5d_adjudication"]["diagnostic_type"]
        for case in reviewed_cases
        if case["phase_5d_adjudication"]["dpo_v1_primary"]
    )
    by_diagnostic_decision = {
        diagnostic: Counter(
            case["phase_5d_adjudication"]["decision"]
            for case in reviewed_cases
            if case["phase_5d_adjudication"]["diagnostic_type"] == diagnostic
        )
        for diagnostic in DIAGNOSTIC_PRIORITY
    }
    by_call_type = Counter(case["call_type"] for case in reviewed_cases)

    phase_5c_population = phase_5c.get("updated_phase_5_population", {})
    prior_high = phase_5c_population.get("combined_high_confidence")
    prior_primary = phase_5c_population.get("combined_dpo_v1_primary")
    phase_5c_decisions = phase_5c.get("adjudication", {}).get(
        "decision_counts", {}
    )
    phase_5b_decisions = phase_5b.get("candidate_decision_counts", {})
    structural_samples = phase_5b.get("structural_candidate_samples")
    mining_samples = candidate_summary.get("mining_population", {}).get(
        "evaluated_samples"
    )
    required_counts = (
        prior_high,
        prior_primary,
        phase_5c_decisions.get("excluded"),
        phase_5c_decisions.get("uncertain"),
        phase_5b_decisions.get("excluded"),
        structural_samples,
        mining_samples,
    )
    if not all(isinstance(value, int) for value in required_counts):
        raise TypeError("Phase 5B/5C summaries contain invalid population counts")
    if phase_5c.get("queue", {}).get("reviewed_samples") + len(queue) != (
        phase_5b_decisions.get("requires_semantic_review")
    ):
        raise ValueError("Phase 5C and Phase 5D queues do not close Phase 5B pending")

    phase_5d_high = decisions["high_confidence"]
    phase_5d_primary = sum(
        case["phase_5d_adjudication"]["dpo_v1_primary"]
        for case in reviewed_cases
    )
    final_high = prior_high + phase_5d_high
    final_primary = prior_primary + phase_5d_primary
    final_excluded = (
        phase_5b_decisions["excluded"]
        + phase_5c_decisions["excluded"]
        + decisions["excluded"]
    )
    final_uncertain = phase_5c_decisions["uncertain"] + decisions["uncertain"]
    if final_high + final_excluded + final_uncertain != structural_samples:
        raise ValueError("Final Phase 5 decision counts do not close")

    if final_high >= 200:
        threshold_band = "at_least_200_ready_for_materialization_and_smoke"
        next_action = "Proceed to DPO-v1 data materialization and smoke testing."
    elif final_high >= 120:
        threshold_band = "120_to_199_review_bucket_diversity"
        next_action = (
            "Review final bucket diversity before choosing a small DPO-v1 run "
            "or expanding the mining pool."
        )
    else:
        threshold_band = "below_120_consider_expanding_to_10k"
        next_action = (
            "Consider expanding the mining pool from 5k to 10k before DPO-v1 "
            "data materialization."
        )

    summary = {
        "analysis": "dpo_v1_train_side_mining",
        "phase": "phase_5d_non_wrong_value_preference_adjudication",
        "scope": "remaining_phase_5b_pending_without_wrong_value",
        "queue": {
            "samples": len(queue),
            "reviewed_samples": len(reviewed_cases),
            "coverage": _rate(len(reviewed_cases), len(queue)),
            "ordered_sample_ids_sha256": _ordered_ids_sha256(queue_ids),
            "review_priority": list(DIAGNOSTIC_PRIORITY),
            "exclusive_priority_diagnostic_counts": {
                diagnostic: diagnostics[diagnostic]
                for diagnostic in DIAGNOSTIC_PRIORITY
            },
            "source_argument_diagnostics_are_non_mutually_exclusive": True,
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
                decision: _rate(decisions[decision], len(reviewed_cases))
                for decision in DECISIONS
            },
            "by_call_type": {
                call_type: by_call_type[call_type] for call_type in CALL_TYPES
            },
            "by_diagnostic": {
                diagnostic: {
                    decision: by_diagnostic_decision[diagnostic][decision]
                    for decision in DECISIONS
                }
                for diagnostic in DIAGNOSTIC_PRIORITY
            },
            "high_confidence_by_diagnostic": {
                diagnostic: high_diagnostics[diagnostic]
                for diagnostic in DIAGNOSTIC_PRIORITY
            },
            "dpo_v1_primary_samples": phase_5d_primary,
            "dpo_v1_primary_by_diagnostic": {
                diagnostic: primary_diagnostics[diagnostic]
                for diagnostic in DIAGNOSTIC_PRIORITY
            },
        },
        "final_5k_inventory": {
            "mining_pool_samples": mining_samples,
            "structural_candidate_samples": structural_samples,
            "all_structural_candidates_adjudicated": True,
            "decision_counts": {
                "high_confidence": final_high,
                "excluded": final_excluded,
                "uncertain": final_uncertain,
            },
            "phase_5c_and_earlier_high_confidence": prior_high,
            "phase_5d_new_high_confidence": phase_5d_high,
            "high_confidence_yield_among_structural_candidates": _rate(
                final_high, structural_samples
            ),
            "high_confidence_yield_among_5k_mining_pool": _rate(
                final_high, mining_samples
            ),
            "dpo_v1_primary_samples": final_primary,
            "phase_5d_new_primary_samples": phase_5d_primary,
        },
        "decision_gate": {
            "policy": {
                "at_least_200": "materialize DPO-v1 data and run smoke test",
                "120_to_199": "inspect bucket diversity before deciding",
                "below_120": "consider expanding mining pool to 10k",
            },
            "final_high_confidence_samples": final_high,
            "threshold_band": threshold_band,
            "next_action": next_action,
            "mining_pool_expanded_in_this_phase": False,
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
        for case in reviewed_cases:
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
    summary = adjudicate_non_wrong_values(
        candidate_cases_path=mining_root / "candidate_cases.jsonl",
        candidate_summary_path=mining_root / "candidate_summary.json",
        filtering_summary_path=mining_root / "filtering_summary.json",
        phase_5c_summary_path=mining_root / "wrong_value_adjudication_summary.json",
        annotations_path=args.annotations.resolve(),
        output_cases_path=mining_root / "non_wrong_value_adjudication.jsonl",
        output_summary_path=mining_root / "non_wrong_value_adjudication_summary.json",
    )
    decisions = summary["adjudication"]["decision_counts"]
    final = summary["final_5k_inventory"]
    print("Phase 5D non-wrong-value preference adjudication complete")
    print(
        f"reviewed={summary['queue']['reviewed_samples']}, "
        f"high_confidence={decisions['high_confidence']}, "
        f"excluded={decisions['excluded']}, uncertain={decisions['uncertain']}"
    )
    print(
        f"final_5k_high_confidence={final['decision_counts']['high_confidence']}, "
        f"final_dpo_v1_primary={final['dpo_v1_primary_samples']}"
    )
    print(f"decision: {summary['decision_gate']['threshold_band']}")
    print(f"cases: {mining_root / 'non_wrong_value_adjudication.jsonl'}")
    print(f"summary: {mining_root / 'non_wrong_value_adjudication_summary.json'}")


if __name__ == "__main__":
    main()

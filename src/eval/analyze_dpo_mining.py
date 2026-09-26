"""Analyze train-side SFT mining results without creating DPO pairs.

Phase 5B applies the frozen evaluator, reuses the Day 3 argument diagnostics,
and records a conservative, auditable preference-quality decision.  Structural
eligibility is never treated as proof that canonical gold is preferable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.eval.analyze_errors import (
        ARGUMENT_ANALYSIS_STATUSES,
        ARGUMENT_DIAGNOSTIC_LABELS,
        CALL_TYPES,
        EVALUATOR_ERROR_LABELS,
        SEMANTIC_BUCKETS,
        _canonical_json,
        _schema_at_path,
        _tool_index,
        _user_query,
        diagnose_argument_sample,
        load_index,
        summarize_argument_cases,
    )
except ModuleNotFoundError:
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))
    from src.eval.analyze_errors import (  # type: ignore[no-redef]
        ARGUMENT_ANALYSIS_STATUSES,
        ARGUMENT_DIAGNOSTIC_LABELS,
        CALL_TYPES,
        EVALUATOR_ERROR_LABELS,
        SEMANTIC_BUCKETS,
        _canonical_json,
        _schema_at_path,
        _tool_index,
        _user_query,
        diagnose_argument_sample,
        load_index,
        summarize_argument_cases,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINING_ROOT = PROJECT_ROOT / "outputs" / "dpo" / "mining_v1"
DEFAULT_CANONICAL = PROJECT_ROOT / "data" / "dpo" / "mining_pool_v1_5k.jsonl"
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "configs"
    / "analysis"
    / "dpo_mining_v1_preference_annotations.json"
)
PRIMARY_BUCKETS = {
    "value_transformation_or_normalization",
    "constraint_or_schema_option_selection",
    "entity_or_literal_extraction",
}
DECISIONS = ("high_confidence", "excluded", "requires_semantic_review")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _normalized_string(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def _representation_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, str) and isinstance(right, str):
        return _normalized_string(left) == _normalized_string(right)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(
            _representation_equivalent(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _representation_equivalent(left[key], right[key]) for key in left
        )
    return left == right


def _scalar_leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _scalar_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _scalar_leaves(child)]
    return [value]


def _literal_in_query(value: Any, query: str) -> bool:
    normalized_query = " ".join(query.casefold().split())
    if value is None:
        return False
    if isinstance(value, bool):
        pattern = rf"(?<!\w){str(value).casefold()}(?!\w)"
        return re.search(pattern, normalized_query) is not None
    if isinstance(value, (int, float)):
        forms = {str(value)}
        if isinstance(value, float) and value.is_integer():
            forms.add(str(int(value)))
        return any(
            re.search(rf"(?<![\d.]){re.escape(form)}(?![\d.])", normalized_query)
            is not None
            for form in forms
        )
    if isinstance(value, str):
        return " ".join(value.casefold().split()) in normalized_query
    leaves = _scalar_leaves(value)
    return bool(leaves) and all(_literal_in_query(leaf, query) for leaf in leaves)


def _schemas_for_diff(
    diff: dict[str, Any],
    tool_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        schema
        for tool in tool_index.get(diff["tool"], [])
        if (schema := _schema_at_path(tool, diff["path"])) is not None
    ]


def _diff_is_default_equivalent(
    diff: dict[str, Any],
    schemas: list[dict[str, Any]],
) -> bool:
    values: list[Any] = []
    if diff["type"] == "missing_argument":
        values = [diff["gold"]]
    elif diff["type"] == "extra_argument":
        values = [diff["pred"]]
    if not values:
        return False
    defaults = [schema["default"] for schema in schemas if "default" in schema]
    return bool(defaults) and all(
        any(_representation_equivalent(value, default) for default in defaults)
        for value in values
    )


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"preference annotations not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path}: annotations must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, annotation in enumerate(value):
        if not isinstance(annotation, dict):
            raise TypeError(f"{path}: annotation {index} must be an object")
        sample_id = annotation.get("sample_id")
        decision = annotation.get("decision")
        reason = annotation.get("reason")
        bucket = annotation.get("semantic_bucket")
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError(f"{path}: annotation {index} has invalid sample_id")
        if decision not in {"high_confidence", "excluded"}:
            raise ValueError(f"{path}: annotation {index} has invalid decision")
        if not isinstance(reason, str) or not reason:
            raise TypeError(f"{path}: annotation {index} has invalid reason")
        if decision == "high_confidence" and bucket not in SEMANTIC_BUCKETS:
            raise ValueError(f"{path}: annotation {index} has invalid bucket")
        if decision == "excluded" and bucket is not None:
            raise ValueError(f"{path}: excluded annotation {index} has a bucket")
        if sample_id in result:
            raise ValueError(f"{path}: duplicate annotation for {sample_id}")
        result[sample_id] = annotation
    return result


def assess_preference(
    case: dict[str, Any],
    canonical: dict[str, Any],
    annotation: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_id = case["sample_id"]
    query = _user_query(canonical.get("messages"), sample_id=sample_id)
    tool_index = _tool_index(canonical.get("tools"), sample_id=sample_id)
    diff_signals = []
    for diff in case["diffs"]:
        schemas = _schemas_for_diff(diff, tool_index)
        equivalent = (
            diff["gold_present"]
            and diff["pred_present"]
            and _representation_equivalent(diff["gold"], diff["pred"])
        )
        default_equivalent = _diff_is_default_equivalent(diff, schemas)
        diff_signals.append(
            {
                "tool": diff["tool"],
                "path": diff["path"],
                "diagnostic": diff["type"],
                "gold_literal_in_query": (
                    diff["gold_present"] and _literal_in_query(diff["gold"], query)
                ),
                "prediction_literal_in_query": (
                    diff["pred_present"] and _literal_in_query(diff["pred"], query)
                ),
                "representation_equivalent": equivalent,
                "default_equivalent": default_equivalent,
                "schema_options": schemas,
            }
        )

    all_diffs_equivalent = bool(diff_signals) and all(
        signal["representation_equivalent"] or signal["default_equivalent"]
        for signal in diff_signals
    )
    all_gold_literal = bool(diff_signals) and all(
        signal["gold_literal_in_query"] for signal in diff_signals
    )

    if annotation is not None:
        decision = annotation["decision"]
        basis = "codex_assisted_case_review"
        reason = annotation["reason"]
        semantic_bucket = annotation.get("semantic_bucket")
    elif all_diffs_equivalent:
        decision = "excluded"
        basis = "deterministic_equivalence_guard"
        reason = "All strict diffs are representation- or schema-default-equivalent."
        semantic_bucket = None
    else:
        decision = "requires_semantic_review"
        basis = "conservative_unresolved"
        reason = (
            "Structure is clean, but canonical gold preference is not established "
            "without semantic review."
        )
        semantic_bucket = None

    return {
        "decision": decision,
        "eligible_for_future_pair_materialization": decision == "high_confidence",
        "basis": basis,
        "reason": reason,
        "semantic_bucket": semantic_bucket,
        "dpo_v1_primary_target": semantic_bucket in PRIMARY_BUCKETS,
        "signals": {
            "all_gold_literals_found_in_query": all_gold_literal,
            "all_diffs_equivalent_or_default_equivalent": all_diffs_equivalent,
            "diffs": diff_signals,
        },
    }


def _validate_inputs(
    canonical: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> None:
    if canonical.keys() != predictions.keys() or canonical.keys() != evaluations.keys():
        raise ValueError("canonical, predictions, and evaluation sample IDs differ")
    for sample_id, evaluation in evaluations.items():
        prediction = predictions[sample_id]
        sample = canonical[sample_id]
        if evaluation.get("gold") != prediction.get("gold"):
            raise ValueError(f"{sample_id}: evaluation and prediction gold differ")
        if evaluation.get("gold") != sample.get("assistant"):
            raise ValueError(f"{sample_id}: evaluation and canonical gold differ")
        if evaluation.get("raw_prediction") != prediction.get("raw_prediction"):
            raise ValueError(f"{sample_id}: evaluation and prediction text differ")
        metrics = evaluation.get("metrics")
        errors = evaluation.get("errors")
        if not isinstance(metrics, dict) or not isinstance(
            metrics.get("full_call_exact"), bool
        ):
            raise TypeError(f"{sample_id}: invalid full_call_exact")
        if not isinstance(errors, list) or errors != metrics.get("errors"):
            raise ValueError(f"{sample_id}: invalid or inconsistent errors")


def analyze_mining_results(
    *,
    summary_path: Path,
    evaluation_path: Path,
    predictions_path: Path,
    canonical_path: Path,
    annotations_path: Path,
    candidate_summary_path: Path,
    candidate_cases_path: Path,
    filtering_summary_path: Path,
    analysis_split_label: str = "train_mining_v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if frozen_summary.get("protocol", {}).get("frozen") is not True:
        raise ValueError(f"{summary_path}: evaluator protocol is not frozen")
    if frozen_summary.get("inputs", {}).get("partial_evaluation") is not False:
        raise ValueError(f"{summary_path}: partial evaluation is not accepted")

    canonical = load_index(canonical_path, label="DPO mining canonical pool")
    predictions = load_index(predictions_path, label="DPO mining predictions")
    evaluations = load_index(evaluation_path, label="DPO mining evaluation")
    _validate_inputs(canonical, predictions, evaluations)
    annotations = load_annotations(annotations_path)

    evaluation_rows = list(evaluations.values())
    failures = [
        row for row in evaluation_rows if row["metrics"]["full_call_exact"] is False
    ]
    error_counts = Counter(error for row in failures for error in row["errors"])
    error_labels = EVALUATOR_ERROR_LABELS + tuple(
        sorted(set(error_counts) - set(EVALUATOR_ERROR_LABELS))
    )
    failures_by_call_type = Counter(
        row["metrics"]["gold_call_kind"] for row in failures
    )

    argument_cases: list[dict[str, Any]] = []
    for evaluation in failures:
        if "argument_mismatch" not in evaluation["errors"]:
            continue
        argument_cases.append(
            diagnose_argument_sample(
                {
                    "sample_id": evaluation["sample_id"],
                    "split": analysis_split_label,
                    "call_type": evaluation["metrics"]["gold_call_kind"],
                    "gold": evaluation["gold"],
                    "sft_prediction": evaluation["parsed_prediction"],
                    "sft_errors": evaluation["errors"],
                }
            )
        )

    structural_candidates = [
        case
        for case in argument_cases
        if not case["mixed_with_other_errors"]
        and case["analysis_status"] == "analyzable"
    ]
    structural_ids = {case["sample_id"] for case in structural_candidates}
    stale_annotations = sorted(annotations.keys() - structural_ids)
    if stale_annotations:
        raise ValueError(
            f"{annotations_path}: annotations outside structural candidates: "
            f"{stale_annotations[:5]}"
        )

    audited_cases: list[dict[str, Any]] = []
    for case in structural_candidates:
        sample_id = case["sample_id"]
        sample = canonical[sample_id]
        relevant_names = {diff["tool"] for diff in case["diffs"]}
        preference = assess_preference(case, sample, annotations.get(sample_id))
        audited_cases.append(
            {
                "record_type": "phase_5b_candidate_audit",
                "source_sample_id": sample_id,
                "source_split": "train",
                "generator": "sft_v1",
                "call_type": case["call_type"],
                "failure_type": "argument_mismatch",
                "evaluator_errors": case["sft_errors"],
                "analysis_status": case["analysis_status"],
                "argument_diagnostics": case["argument_diagnostics"],
                "argument_diffs": case["diffs"],
                "matching": case["matching"],
                "user_query": _user_query(sample["messages"], sample_id=sample_id),
                "relevant_tools": [
                    tool
                    for tool in sample["tools"]
                    if tool.get("function", {}).get("name") in relevant_names
                ],
                "canonical_gold": case["gold"],
                "sft_prediction": case["sft_prediction"],
                "sft_raw_prediction": evaluations[sample_id]["raw_prediction"],
                "preference_filter": preference,
                "preference_pair_materialized": False,
            }
        )

    decisions = Counter(
        case["preference_filter"]["decision"] for case in audited_cases
    )
    bases = Counter(case["preference_filter"]["basis"] for case in audited_cases)
    buckets = Counter(
        case["preference_filter"]["semantic_bucket"]
        for case in audited_cases
        if case["preference_filter"]["decision"] == "high_confidence"
    )
    high_by_call_type = Counter(
        case["call_type"]
        for case in audited_cases
        if case["preference_filter"]["decision"] == "high_confidence"
    )

    def decision_breakdown(decision: str) -> dict[str, Any]:
        selected = [
            case
            for case in audited_cases
            if case["preference_filter"]["decision"] == decision
        ]
        call_types = Counter(case["call_type"] for case in selected)
        diagnostics = Counter(
            diagnostic
            for case in selected
            for diagnostic in case["argument_diagnostics"]
        )
        return {
            "samples": len(selected),
            "by_call_type": {
                call_type: call_types[call_type]
                for call_type in CALL_TYPES
                if call_types[call_type]
            },
            "by_argument_diagnostic": {
                diagnostic: diagnostics[diagnostic]
                for diagnostic in ARGUMENT_DIAGNOSTIC_LABELS
                if diagnostics[diagnostic]
            },
            "diagnostics_are_non_mutually_exclusive": True,
        }

    candidate_summary = {
        "analysis": "dpo_v1_train_side_mining",
        "phase": "phase_5b_mining_result_analysis",
        "frozen_evaluator_outputs_only": True,
        "preference_pairs_generated": False,
        "inputs": {
            "summary": str(summary_path.resolve()),
            "evaluation": str(evaluation_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "canonical": str(canonical_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "evaluation_sha256": _sha256(evaluation_path),
            "predictions_sha256": _sha256(predictions_path),
            "canonical_sha256": _sha256(canonical_path),
        },
        "mining_population": {
            "evaluated_samples": len(evaluation_rows),
            "full_call_success_samples": len(evaluation_rows) - len(failures),
            "failure_samples": len(failures),
            "failure_rate_among_all_samples": _rate(
                len(failures), len(evaluation_rows)
            ),
            "frozen_full_call_success_rate": frozen_summary["metrics"]["primary"][
                "full_call_success_rate"
            ],
        },
        "failure_distribution": {
            "error_occurrences": sum(error_counts.values()),
            "multi_label_failure_samples": sum(
                len(row["errors"]) > 1 for row in failures
            ),
            "errors_are_non_mutually_exclusive": True,
            "error_counts": {label: error_counts[label] for label in error_labels},
            "error_rates_among_failures": {
                label: error_counts[label] / len(failures) for label in error_labels
            },
            "by_call_type": {
                call_type: {
                    "failure_samples": failures_by_call_type[call_type],
                    "failure_rate_within_call_type": _rate(
                        failures_by_call_type[call_type],
                        sum(
                            row["metrics"]["gold_call_kind"] == call_type
                            for row in evaluation_rows
                        ),
                    ),
                }
                for call_type in CALL_TYPES
                if failures_by_call_type[call_type]
            },
        },
        "argument_diagnostics": {
            **summarize_argument_cases(argument_cases),
            "clean_analyzable_candidate_samples": len(structural_candidates),
        },
        "candidate_cases": str(candidate_cases_path.resolve()),
    }

    filtering_summary = {
        "analysis": "dpo_v1_train_side_mining",
        "phase": "phase_5b_preference_quality_filtering",
        "structural_candidate_samples": len(structural_candidates),
        "candidate_decision_counts": {
            decision: decisions[decision] for decision in DECISIONS
        },
        "candidate_decision_breakdowns": {
            decision: decision_breakdown(decision) for decision in DECISIONS
        },
        "decision_basis_counts": dict(sorted(bases.items())),
        "review_annotations": {
            "source": str(annotations_path.resolve()),
            "provenance": "codex_assisted_case_by_case_review_not_human_ground_truth",
            "reviewed_samples": len(annotations),
            "coverage_among_structural_candidates": _rate(
                len(annotations), len(structural_candidates)
            ),
        },
        "high_confidence_candidates": {
            "samples": decisions["high_confidence"],
            "by_call_type": {
                call_type: high_by_call_type[call_type]
                for call_type in CALL_TYPES
                if high_by_call_type[call_type]
            },
            "by_semantic_bucket": {
                bucket: buckets[bucket] for bucket in SEMANTIC_BUCKETS
            },
            "dpo_v1_primary_target_samples": sum(
                buckets[bucket] for bucket in PRIMARY_BUCKETS
            ),
        },
        "filtering_contract": {
            "strict_mismatch_is_not_preference_proof": True,
            "high_confidence_requires_case_review": True,
            "deterministic_auto_exclusion": (
                "all diffs are representation- or schema-default-equivalent"
            ),
            "unresolved_cases_remain_pending": True,
            "excluded_categories": [
                "semantic or representation equivalence",
                "query underdetermination",
                "gold over-specificity or gold issue",
                "opaque ID not derivable from query",
                "multiple reasonable schema-valid representations",
            ],
        },
        "ready_for_dpo_pair_materialization": False,
        "preference_pairs_generated": False,
        "dpo_train_written": False,
        "next_decision": (
            "Continue semantic review of pending structural candidates before "
            "deciding whether to expand the mining pool."
        ),
    }

    candidate_cases_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_cases_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in audited_cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    _write_json(candidate_summary_path, candidate_summary)
    _write_json(filtering_summary_path, filtering_summary)
    return candidate_summary, filtering_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining-root", type=Path, default=DEFAULT_MINING_ROOT)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--analysis-split-label",
        default="train_mining_v1",
        help="Provenance label stored in argument diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mining_root = args.mining_root.resolve()
    candidate_summary, filtering_summary = analyze_mining_results(
        summary_path=mining_root / "summary.json",
        evaluation_path=mining_root / "evaluation.jsonl",
        predictions_path=mining_root / "predictions.jsonl",
        canonical_path=args.canonical.resolve(),
        annotations_path=args.annotations.resolve(),
        candidate_summary_path=mining_root / "candidate_summary.json",
        candidate_cases_path=mining_root / "candidate_cases.jsonl",
        filtering_summary_path=mining_root / "filtering_summary.json",
        analysis_split_label=args.analysis_split_label,
    )
    population = candidate_summary["mining_population"]
    argument = candidate_summary["argument_diagnostics"]
    decisions = filtering_summary["candidate_decision_counts"]
    print("Phase 5B mining analysis complete")
    print(
        f"samples={population['evaluated_samples']}, "
        f"failures={population['failure_samples']}, "
        f"argument_mismatches={argument['argument_mismatch_samples']}"
    )
    print(
        f"clean_analyzable={argument['clean_analyzable_candidate_samples']}, "
        f"high_confidence={decisions['high_confidence']}, "
        f"excluded={decisions['excluded']}, "
        f"pending_review={decisions['requires_semantic_review']}"
    )
    print(f"candidate summary: {mining_root / 'candidate_summary.json'}")
    print(f"candidate cases: {mining_root / 'candidate_cases.jsonl'}")
    print(f"filtering summary: {mining_root / 'filtering_summary.json'}")


if __name__ == "__main__":
    main()

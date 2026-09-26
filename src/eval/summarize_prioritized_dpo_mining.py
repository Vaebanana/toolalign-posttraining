"""Summarize verifier-backed yield from AUTO_ACCEPT-prioritized mining.

This phase consumes frozen evaluator and structural-diagnostic outputs, applies
three-way difference-level triage, and stops before preference-pair or trainer
materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.eval.dpo_groundability import (
        GROUNDABILITY_VERSION,
        triage_candidate_difference_groundability,
    )
except ModuleNotFoundError:
    from dpo_groundability import (  # type: ignore[no-redef]
        GROUNDABILITY_VERSION,
        triage_candidate_difference_groundability,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINING_ROOT = PROJECT_ROOT / "outputs" / "dpo" / "mining_v3_auto_accept"
DEFAULT_BASELINE_POOL = (
    PROJECT_ROOT / "outputs" / "dpo" / "dpo_v1_grounded_primary_pool.json"
)
DEFAULT_BASELINE_MINING_ROOTS = (
    PROJECT_ROOT / "outputs" / "dpo" / "mining_v1",
    PROJECT_ROOT / "outputs" / "dpo" / "mining_v2",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected an object")
        records.append(value)
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def summarize_prioritized_mining(
    *,
    mining_root: Path,
    baseline_pool_path: Path,
    baseline_mining_roots: list[Path],
) -> dict[str, Any]:
    summary_path = mining_root / "summary.json"
    candidate_summary_path = mining_root / "candidate_summary.json"
    filtering_summary_path = mining_root / "filtering_summary.json"
    cases_path = mining_root / "candidate_cases.jsonl"
    for path in (
        summary_path,
        candidate_summary_path,
        filtering_summary_path,
        cases_path,
        baseline_pool_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")

    evaluation_summary = _read_json(summary_path)
    candidate_summary = _read_json(candidate_summary_path)
    filtering_summary = _read_json(filtering_summary_path)
    baseline_pool = _read_json(baseline_pool_path)
    baseline_candidate_summaries = [
        _read_json(root / "candidate_summary.json")
        for root in baseline_mining_roots
    ]
    cases = _read_jsonl(cases_path)
    structural_count = candidate_summary["argument_diagnostics"][
        "clean_analyzable_candidate_samples"
    ]
    if len(cases) != structural_count:
        raise ValueError("candidate JSONL count differs from candidate summary")

    decisions = Counter()
    reasons = Counter()
    by_call_type: dict[str, Counter[str]] = {}
    by_diagnostic: dict[str, Counter[str]] = {}
    evidence = Counter()
    output_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in cases:
        sample_id = case.get("source_sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError("candidate has no source_sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate candidate sample ID: {sample_id}")
        seen_ids.add(sample_id)
        triage = triage_candidate_difference_groundability(case)
        decision = triage["decision"]
        reason = triage["reason"]
        decisions[decision] += 1
        reasons[reason] += 1
        call_type = case["call_type"]
        by_call_type.setdefault(call_type, Counter())[decision] += 1
        for diagnostic in case["argument_diagnostics"]:
            by_diagnostic.setdefault(diagnostic, Counter())[decision] += 1
        groundability = triage.get("groundability")
        if isinstance(groundability, dict):
            evidence.update(groundability.get("evidence_counts", {}))
        output_cases.append(
            {
                "record_type": "phase_5f_difference_groundability_triage",
                "source_sample_id": sample_id,
                "source_split": case.get("source_split"),
                "call_type": call_type,
                "argument_diagnostics": case["argument_diagnostics"],
                "argument_diffs": case["argument_diffs"],
                "user_query": case["user_query"],
                "relevant_tools": case["relevant_tools"],
                "canonical_gold": case["canonical_gold"],
                "sft_prediction": case["sft_prediction"],
                "sft_raw_prediction": case["sft_raw_prediction"],
                "difference_groundability_triage": triage,
                "preference_pair_materialized": False,
            }
        )

    if sum(decisions.values()) != structural_count:
        raise ValueError("difference triage does not close to structural candidates")
    baseline_count = baseline_pool["inventory"]["grounded_primary_candidates"]
    baseline_rollouts = sum(
        item["mining_population"]["evaluated_samples"]
        for item in baseline_candidate_summaries
    )
    baseline_failures = sum(
        item["mining_population"]["failure_samples"]
        for item in baseline_candidate_summaries
    )
    baseline_structural = sum(
        item["argument_diagnostics"]["clean_analyzable_candidate_samples"]
        for item in baseline_candidate_summaries
    )
    pass_count = decisions["PASS"]
    new_rollouts = candidate_summary["mining_population"]["evaluated_samples"]
    baseline_yield = baseline_count / baseline_rollouts
    new_yield = pass_count / new_rollouts
    combined_count = baseline_count + pass_count

    result = {
        "artifact": "phase_5f_auto_accept_prioritized_mining_yield",
        "phase": "phase_5f_auto_accept_prioritized_rollout",
        "status": "difference_triage_complete_no_pairs",
        "policy": {
            "prompt_level_auto_accept_is_rollout_priority_only": True,
            "difference_decisions": ["PASS", "REJECT", "UNKNOWN"],
            "pass_definition": (
                "Every observed preference difference is verifier-backed and "
                "no implemented equivalence or ambiguity guard fired."
            ),
            "reject_definition": (
                "A deterministic equivalence/default guard or obvious opaque "
                "external mapping invalidates direct preference use."
            ),
            "unknown_definition": (
                "Rules abstain; optional extras, nested alternatives, and other "
                "semantic gray cases are not judged."
            ),
            "manual_or_llm_review_limited_to_unknown": True,
            "groundability_version": GROUNDABILITY_VERSION,
        },
        "inputs": {
            "summary": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "candidate_summary": str(candidate_summary_path.resolve()),
            "candidate_summary_sha256": _sha256(candidate_summary_path),
            "filtering_summary": str(filtering_summary_path.resolve()),
            "filtering_summary_sha256": _sha256(filtering_summary_path),
            "candidate_cases": str(cases_path.resolve()),
            "candidate_cases_sha256": _sha256(cases_path),
            "random_mining_grounded_pool": str(baseline_pool_path.resolve()),
            "random_mining_grounded_pool_sha256": _sha256(baseline_pool_path),
            "random_mining_candidate_summaries": {
                str((root / "candidate_summary.json").resolve()): _sha256(
                    root / "candidate_summary.json"
                )
                for root in baseline_mining_roots
            },
        },
        "funnel": {
            "rollout_samples": new_rollouts,
            "sft_failures": candidate_summary["mining_population"]["failure_samples"],
            "argument_failure_samples": candidate_summary["argument_diagnostics"][
                "argument_mismatch_samples"
            ],
            "clean_analyzable_structural_candidates": structural_count,
            "difference_gate_pass": pass_count,
            "difference_gate_reject": decisions["REJECT"],
            "difference_gate_unknown": decisions["UNKNOWN"],
            "difference_gate_closure_verified": sum(decisions.values())
            == structural_count,
        },
        "difference_triage": {
            "decision_counts": {
                decision: decisions[decision]
                for decision in ("PASS", "REJECT", "UNKNOWN")
            },
            "reason_counts": dict(sorted(reasons.items())),
            "by_call_type": {
                key: {
                    decision: value[decision]
                    for decision in ("PASS", "REJECT", "UNKNOWN")
                }
                for key, value in sorted(by_call_type.items())
            },
            "by_argument_diagnostic": {
                key: {
                    decision: value[decision]
                    for decision in ("PASS", "REJECT", "UNKNOWN")
                }
                for key, value in sorted(by_diagnostic.items())
            },
            "diagnostics_are_non_mutually_exclusive": True,
            "grounding_evidence_occurrences": dict(sorted(evidence.items())),
        },
        "yield_comparison": {
            "random_mining_v1_v2": {
                "rollout_samples": baseline_rollouts,
                "sft_failures": baseline_failures,
                "clean_analyzable_structural_candidates": baseline_structural,
                "grounded_primary_candidates": baseline_count,
                "yield": _rate(baseline_count, baseline_rollouts),
                "failure_rate": _rate(baseline_failures, baseline_rollouts),
                "structural_candidate_rate": _rate(
                    baseline_structural, baseline_rollouts
                ),
                "grounded_rate_among_structural_candidates": _rate(
                    baseline_count, baseline_structural
                ),
            },
            "auto_accept_prioritized_v3": {
                "rollout_samples": new_rollouts,
                "sft_failures": candidate_summary["mining_population"][
                    "failure_samples"
                ],
                "clean_analyzable_structural_candidates": structural_count,
                "verifier_backed_difference_pass_candidates": pass_count,
                "yield": _rate(pass_count, new_rollouts),
                "failure_rate": _rate(
                    candidate_summary["mining_population"]["failure_samples"],
                    new_rollouts,
                ),
                "structural_candidate_rate": _rate(structural_count, new_rollouts),
                "pass_rate_among_structural_candidates": _rate(
                    pass_count, structural_count
                ),
            },
            "yield_uplift_ratio": new_yield / baseline_yield,
            "conditional_pass_uplift_ratio": (
                (pass_count / structural_count)
                / (baseline_count / baseline_structural)
            ),
            "observed_mechanism": (
                "AUTO_ACCEPT prioritization selected a substantially easier "
                "population for SFT-v1, reducing the supply of on-policy "
                "failures before the final gate."
            ),
            "comparison_caveat": (
                "The baseline includes model-assisted semantic adjudication; "
                "the new numerator is rule-verifier PASS and is not a human "
                "accuracy estimate."
            ),
        },
        "inventory_decision": {
            "existing_grounded_seed_candidates": baseline_count,
            "new_verifier_backed_pass_candidates": pass_count,
            "combined_candidate_inventory": combined_count,
            "target_range": [120, 150],
            "minimum_target_reached": combined_count >= 120,
            "recommended_action": (
                "stop deterministic mining and audit/materialize the combined pool"
                if pass_count >= 70
                else "stop deterministic mining and choose a supplemental source"
                if pass_count < 30
                else "review bucket diversity before another mining decision"
            ),
        },
        "operations": {
            "preference_pairs_generated": False,
            "dpo_train_written": False,
            "trainer_created": False,
        },
        "frozen_evaluator_summary": evaluation_summary,
        "phase_5b_filtering_summary": filtering_summary,
    }
    output_cases_path = mining_root / "groundability_cases.jsonl"
    output_summary_path = mining_root / "groundability_summary.json"
    with output_cases_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in output_cases:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with output_summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining-root", type=Path, default=DEFAULT_MINING_ROOT)
    parser.add_argument("--baseline-pool", type=Path, default=DEFAULT_BASELINE_POOL)
    parser.add_argument(
        "--baseline-mining-root",
        action="append",
        type=Path,
        dest="baseline_mining_roots",
        help="Repeat for random-mining baselines; defaults to mining_v1 and v2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_prioritized_mining(
        mining_root=args.mining_root.resolve(),
        baseline_pool_path=args.baseline_pool.resolve(),
        baseline_mining_roots=[
            path.resolve()
            for path in (
                args.baseline_mining_roots or DEFAULT_BASELINE_MINING_ROOTS
            )
        ],
    )
    funnel = result["funnel"]
    comparison = result["yield_comparison"]
    print("Phase 5F prioritized mining groundability triage complete")
    print(
        "funnel: "
        f"{funnel['rollout_samples']} -> {funnel['sft_failures']} -> "
        f"{funnel['argument_failure_samples']} -> "
        f"{funnel['clean_analyzable_structural_candidates']} -> "
        f"PASS {funnel['difference_gate_pass']} / "
        f"REJECT {funnel['difference_gate_reject']} / "
        f"UNKNOWN {funnel['difference_gate_unknown']}"
    )
    print(f"yield_uplift_ratio={comparison['yield_uplift_ratio']:.4f}")
    print(f"summary={args.mining_root.resolve() / 'groundability_summary.json'}")
    print(f"cases={args.mining_root.resolve() / 'groundability_cases.jsonl'}")
    print("preference_pairs_generated=false")


if __name__ == "__main__":
    main()

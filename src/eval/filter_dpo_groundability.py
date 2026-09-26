"""Apply the final model-visible groundability gate to DPO primary candidates.

This step preserves earlier model-assisted adjudications as provenance, but a
candidate remains eligible only when every preference-bearing canonical value
is grounded in the user query plus the exposed tool schema.  It does not create
DPO pairs or trainer configuration.
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
        assess_case_groundability,
    )
except ModuleNotFoundError:
    from dpo_groundability import (  # type: ignore[no-redef]
        GROUNDABILITY_VERSION,
        assess_case_groundability,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINING_ROOTS = (
    PROJECT_ROOT / "outputs" / "dpo" / "mining_v1",
    PROJECT_ROOT / "outputs" / "dpo" / "mining_v2",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dpo" / "dpo_v1_grounded_primary_pool.json"
DEFAULT_HUMAN_AUDIT = (
    PROJECT_ROOT / "configs" / "analysis" / "dpo_v1_primary_human_audit_20.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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


def _candidate_annotation(
    record: dict[str, Any], stage: str
) -> tuple[dict[str, Any], str, str]:
    if stage == "phase_5b":
        annotation = record["preference_filter"]
        primary = annotation.get("dpo_v1_primary_target")
        bucket = annotation.get("semantic_bucket")
        reason = annotation.get("reason")
    elif stage == "phase_5c":
        annotation = record["phase_5c_adjudication"]
        primary = annotation.get("dpo_v1_primary")
        bucket = annotation.get("semantic_bucket")
        reason = annotation.get("preference_reason")
    elif stage == "phase_5d":
        annotation = record["phase_5d_adjudication"]
        primary = annotation.get("dpo_v1_primary")
        diagnostic = annotation.get("diagnostic_type")
        bucket = (
            diagnostic
            if diagnostic in {"missing_argument", "extra_argument"}
            else "other_primary"
        )
        reason = annotation.get("semantic_reason")
    else:
        raise ValueError(f"unexpected stage: {stage}")
    if annotation.get("decision") != "high_confidence" or primary is not True:
        raise ValueError("record is not a pre-groundability primary candidate")
    if not isinstance(bucket, str) or not bucket:
        raise TypeError("primary candidate has no bucket")
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("primary candidate has no preference reason")
    return annotation, bucket, reason


def collect_primary_candidates(
    mining_roots: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sources = (
        ("phase_5b", "candidate_cases.jsonl", "preference_filter", "dpo_v1_primary_target"),
        ("phase_5c", "wrong_value_adjudication.jsonl", "phase_5c_adjudication", "dpo_v1_primary"),
        ("phase_5d", "non_wrong_value_adjudication.jsonl", "phase_5d_adjudication", "dpo_v1_primary"),
    )
    candidates: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for mining_root in mining_roots:
        batch = mining_root.name
        for stage, filename, annotation_key, primary_key in sources:
            path = mining_root / filename
            input_hashes[str(path.resolve())] = _sha256(path)
            for record in _read_jsonl(path):
                annotation = record.get(annotation_key, {})
                if (
                    annotation.get("decision") == "high_confidence"
                    and annotation.get(primary_key) is True
                ):
                    _, bucket, reason = _candidate_annotation(record, stage)
                    candidates.append(
                        {
                            "source_batch": batch,
                            "source_stage": stage,
                            "semantic_bucket": bucket,
                            "model_assisted_preference_reason": reason,
                            "record": record,
                        }
                    )
    sample_ids = [item["record"].get("source_sample_id") for item in candidates]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
        raise TypeError("primary inventory contains invalid sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("primary inventory contains duplicate sample IDs")
    return candidates, input_hashes


def _human_audit_regression(
    path: Path, decisions: dict[str, bool]
) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    cases = audit.get("cases")
    if not isinstance(cases, list):
        raise TypeError(f"{path}: cases must be an array")
    counts = Counter()
    mismatches: list[str] = []
    for case in cases:
        sample_id = case.get("source_sample_id")
        human = case.get("human_review", {}).get("decision")
        if human not in {"PASS", "FAIL", "UNSURE"}:
            raise ValueError(f"{path}: {sample_id} has no completed human decision")
        if sample_id not in decisions:
            raise ValueError(f"{path}: {sample_id} is outside the primary inventory")
        gate_passed = decisions[sample_id]
        expected_pass = human == "PASS"
        counts[human] += 1
        if gate_passed != expected_pass:
            mismatches.append(sample_id)
    return {
        "source": str(path.resolve()),
        "source_sha256": _sha256(path),
        "reviewed_samples": len(cases),
        "human_decision_counts": {
            decision: counts[decision] for decision in ("PASS", "FAIL", "UNSURE")
        },
        "expected_rule": "groundability pass iff human decision is PASS",
        "matching_samples": len(cases) - len(mismatches),
        "mismatching_samples": len(mismatches),
        "mismatching_sample_ids": mismatches,
        "regression_passed": not mismatches,
    }


def filter_grounded_candidates(
    *,
    mining_roots: list[Path],
    human_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidates, input_hashes = collect_primary_candidates(mining_roots)
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    by_batch_before = Counter()
    by_batch_after = Counter()
    by_stage_before = Counter()
    by_stage_after = Counter()
    by_bucket_before = Counter()
    by_bucket_after = Counter()
    evidence_counts = Counter()
    exclusion_reasons = Counter()
    decisions: dict[str, bool] = {}

    for item in candidates:
        record = item["record"]
        sample_id = record["source_sample_id"]
        groundability = assess_case_groundability(record)
        decisions[sample_id] = groundability["passed"]
        by_batch_before[item["source_batch"]] += 1
        by_stage_before[item["source_stage"]] += 1
        by_bucket_before[item["semantic_bucket"]] += 1
        evidence_counts.update(groundability["evidence_counts"])

        common = {
            "source_sample_id": sample_id,
            "source_split": record.get("source_split"),
            "source_batch": item["source_batch"],
            "source_stage": item["source_stage"],
            "call_type": record.get("call_type"),
            "semantic_bucket": item["semantic_bucket"],
            "pre_groundability_decision": "high_confidence",
            "pre_groundability_dpo_v1_primary": True,
            "model_assisted_preference_reason": item[
                "model_assisted_preference_reason"
            ],
            "groundability": groundability,
            "preference_pair_materialized": False,
        }
        if groundability["passed"]:
            by_batch_after[item["source_batch"]] += 1
            by_stage_after[item["source_stage"]] += 1
            by_bucket_after[item["semantic_bucket"]] += 1
            accepted.append(
                {
                    **common,
                    "record_type": "grounded_dpo_primary_candidate",
                    "final_preference_decision": "grounded_high_confidence",
                    "dpo_v1_primary": True,
                    "user_query": record.get("user_query"),
                    "relevant_tools": record.get("relevant_tools"),
                    "canonical_gold": record.get("canonical_gold"),
                    "sft_prediction": record.get("sft_prediction"),
                    "sft_raw_prediction": record.get("sft_raw_prediction"),
                    "argument_diagnostics": record.get("argument_diagnostics"),
                    "argument_diffs": record.get("argument_diffs"),
                }
            )
        else:
            reason = (
                "opaque_or_external_mapping"
                if any(
                    leaf.get("looks_opaque")
                    for diff in groundability["diff_assessments"]
                    for leaf in diff["ungrounded_leaves"]
                )
                else "not_deterministically_grounded_in_query_and_schema"
            )
            exclusion_reasons[reason] += 1
            excluded.append(
                {
                    **common,
                    "record_type": "groundability_exclusion",
                    "final_preference_decision": "excluded_ungrounded",
                    "dpo_v1_primary": False,
                    "exclusion_reason": reason,
                }
            )

    audit_regression = _human_audit_regression(human_audit_path, decisions)
    if not audit_regression["regression_passed"]:
        raise ValueError(
            "groundability gate does not reproduce the completed human audit: "
            f"{audit_regression['mismatching_sample_ids']}"
        )
    selected_ids = [record["source_sample_id"] for record in accepted]
    output = {
        "artifact": "dpo_v1_grounded_primary_candidate_pool",
        "phase": "phase_5c_groundability_revision",
        "groundability_version": GROUNDABILITY_VERSION,
        "status": "candidate_pool_only_not_dpo_pairs",
        "preference_pairs_generated": False,
        "dpo_train_written": False,
        "trainer_created": False,
        "policy": {
            "model_visible_evidence_only": True,
            "all_preference_differences_must_be_grounded": True,
            "accepted_evidence": [
                "query literal",
                "deterministic query plus schema transformation",
                "explicit schema option or mapping",
                "schema-declared ISO standard mapping",
                "schema-valid omission of an extra argument already judged harmful",
            ],
            "rejected_evidence": [
                "external entity-to-ID, UUID, URI, ticker, or private-code knowledge",
                "schema default used as an entity mapping",
                "undocumented numeric option encoding",
                "invented granularity or unstated transformation premise",
                "empty-string override behavior not documented by the schema",
            ],
        },
        "inputs": {
            "mining_roots": [str(path.resolve()) for path in mining_roots],
            "files_sha256": input_hashes,
        },
        "inventory": {
            "pre_groundability_primary_candidates": len(candidates),
            "grounded_primary_candidates": len(accepted),
            "excluded_ungrounded_candidates": len(excluded),
            "retention_rate": _rate(len(accepted), len(candidates)),
            "ordered_grounded_sample_ids_sha256": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
            "by_batch": {
                batch: {
                    "before": by_batch_before[batch],
                    "grounded": by_batch_after[batch],
                    "excluded": by_batch_before[batch] - by_batch_after[batch],
                }
                for batch in sorted(by_batch_before)
            },
            "by_stage": {
                stage: {
                    "before": by_stage_before[stage],
                    "grounded": by_stage_after[stage],
                    "excluded": by_stage_before[stage] - by_stage_after[stage],
                }
                for stage in sorted(by_stage_before)
            },
            "by_semantic_bucket": {
                bucket: {
                    "before": by_bucket_before[bucket],
                    "grounded": by_bucket_after[bucket],
                    "excluded": by_bucket_before[bucket] - by_bucket_after[bucket],
                }
                for bucket in sorted(by_bucket_before)
            },
            "grounding_evidence_occurrences": dict(sorted(evidence_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        },
        "human_audit_regression": audit_regression,
        "decision_gate": {
            "previous_208_candidate_gate_invalidated": True,
            "minimum_grounded_pairs_for_formal_dpo_v1": 150,
            "grounded_candidates": len(accepted),
            "threshold_met": len(accepted) >= 150,
            "ready_for_pair_materialization": False,
            "reason": (
                "The corrected grounded inventory must be assessed before any "
                "preference-pair materialization; this phase intentionally writes "
                "no DPO pairs."
            ),
        },
        "grounded_candidates": accepted,
        "excluded_candidates": excluded,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mining-root",
        action="append",
        type=Path,
        dest="mining_roots",
        help="Repeat for each mining batch; defaults to mining_v1 and mining_v2.",
    )
    parser.add_argument("--human-audit", type=Path, default=DEFAULT_HUMAN_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.mining_roots or list(DEFAULT_MINING_ROOTS)
    result = filter_grounded_candidates(
        mining_roots=[path.resolve() for path in roots],
        human_audit_path=args.human_audit.resolve(),
        output_path=args.output.resolve(),
    )
    inventory = result["inventory"]
    print("DPO primary groundability filtering complete")
    print(
        f"before={inventory['pre_groundability_primary_candidates']}, "
        f"grounded={inventory['grounded_primary_candidates']}, "
        f"excluded={inventory['excluded_ungrounded_candidates']}"
    )
    print(
        "human_audit_regression="
        f"{result['human_audit_regression']['matching_samples']}/"
        f"{result['human_audit_regression']['reviewed_samples']}"
    )
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()


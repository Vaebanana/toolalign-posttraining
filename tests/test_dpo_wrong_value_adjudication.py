"""Tests for Phase 5C semantic preference adjudication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.eval.adjudicate_dpo_wrong_values import adjudicate_wrong_values


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def candidate(sample_id: str, call_type: str = "single") -> dict[str, object]:
    return {
        "record_type": "phase_5b_candidate_audit",
        "source_sample_id": sample_id,
        "call_type": call_type,
        "argument_diagnostics": ["wrong_value"],
        "preference_filter": {"decision": "requires_semantic_review"},
        "preference_pair_materialized": False,
    }


class DpoWrongValueAdjudicationTests(unittest.TestCase):
    def test_complete_queue_is_materialized_without_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "candidate_cases.jsonl"
            filtering_path = root / "filtering_summary.json"
            annotations_path = root / "annotations.json"
            output_cases = root / "wrong_value_adjudication.jsonl"
            output_summary = root / "wrong_value_adjudication_summary.json"
            write_jsonl(cases_path, [candidate("a"), candidate("b", "multi")])
            write_json(
                filtering_path,
                {
                    "structural_candidate_samples": 10,
                    "candidate_decision_counts": {
                        "high_confidence": 3,
                        "excluded": 2,
                        "requires_semantic_review": 5,
                    },
                    "high_confidence_candidates": {
                        "dpo_v1_primary_target_samples": 2
                    },
                },
            )
            queue_hash = hashlib.sha256("a\nb".encode()).hexdigest()
            write_json(
                annotations_path,
                {
                    "annotation_provenance": (
                        "codex_assisted_case_by_case_review_not_human_ground_truth"
                    ),
                    "queue_sample_count": 2,
                    "queue_sample_ids_sha256": queue_hash,
                    "annotations": [
                        {
                            "sample_id": "a",
                            "decision": "high_confidence",
                            "semantic_bucket": "entity_or_literal_extraction",
                            "preference_reason": "The literal is explicit.",
                            "dpo_v1_primary": True,
                        },
                        {
                            "sample_id": "b",
                            "decision": "excluded",
                            "semantic_bucket": "ambiguous_or_gold_issue",
                            "preference_reason": "Both values are reasonable.",
                            "dpo_v1_primary": False,
                        },
                    ],
                },
            )

            summary = adjudicate_wrong_values(
                candidate_cases_path=cases_path,
                filtering_summary_path=filtering_path,
                annotations_path=annotations_path,
                output_cases_path=output_cases,
                output_summary_path=output_summary,
            )
            records = [
                json.loads(line)
                for line in output_cases.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["queue"]["reviewed_samples"], 2)
        self.assertEqual(
            summary["adjudication"]["decision_counts"],
            {"high_confidence": 1, "excluded": 1, "uncertain": 0},
        )
        self.assertEqual(
            summary["updated_phase_5_population"]["combined_high_confidence"], 4
        )
        self.assertFalse(summary["outputs"]["preference_pairs_generated"])
        self.assertTrue(
            all(not record["preference_pair_materialized"] for record in records)
        )

    def test_incomplete_annotations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "candidate_cases.jsonl"
            filtering_path = root / "filtering_summary.json"
            annotations_path = root / "annotations.json"
            write_jsonl(cases_path, [candidate("a"), candidate("b")])
            write_json(
                filtering_path,
                {
                    "structural_candidate_samples": 2,
                    "candidate_decision_counts": {
                        "high_confidence": 0,
                        "excluded": 0,
                        "requires_semantic_review": 2,
                    },
                    "high_confidence_candidates": {
                        "dpo_v1_primary_target_samples": 0
                    },
                },
            )
            write_json(
                annotations_path,
                {
                    "annotation_provenance": (
                        "codex_assisted_case_by_case_review_not_human_ground_truth"
                    ),
                    "queue_sample_count": 2,
                    "queue_sample_ids_sha256": hashlib.sha256(
                        "a\nb".encode()
                    ).hexdigest(),
                    "annotations": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "cover the ordered"):
                adjudicate_wrong_values(
                    candidate_cases_path=cases_path,
                    filtering_summary_path=filtering_path,
                    annotations_path=annotations_path,
                    output_cases_path=root / "cases.jsonl",
                    output_summary_path=root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase 5D non-wrong-value preference adjudication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.eval.adjudicate_dpo_non_wrong_values import (
    adjudicate_non_wrong_values,
)


PROVENANCE = "codex_assisted_case_by_case_review_not_human_ground_truth"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def candidate(
    sample_id: str,
    diagnostic: str,
    *,
    pending: bool = True,
) -> dict[str, object]:
    return {
        "source_sample_id": sample_id,
        "call_type": "single",
        "argument_diagnostics": [diagnostic],
        "preference_filter": {
            "decision": "requires_semantic_review" if pending else "excluded"
        },
        "preference_pair_materialized": False,
    }


class DpoNonWrongValueAdjudicationTests(unittest.TestCase):
    def test_final_inventory_closes_without_materializing_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "candidate_cases.jsonl"
            candidate_summary_path = root / "candidate_summary.json"
            filtering_path = root / "filtering_summary.json"
            phase_5c_path = root / "phase_5c.json"
            annotations_path = root / "annotations.json"
            output_cases_path = root / "cases.jsonl"
            output_summary_path = root / "summary.json"
            write_jsonl(
                cases_path,
                [
                    candidate("extra", "extra_argument"),
                    candidate("missing", "missing_argument"),
                    candidate("wrong", "wrong_value"),
                    candidate("old_excluded", "extra_argument", pending=False),
                ],
            )
            write_json(
                candidate_summary_path,
                {"mining_population": {"evaluated_samples": 10}},
            )
            write_json(
                filtering_path,
                {
                    "structural_candidate_samples": 4,
                    "candidate_decision_counts": {
                        "high_confidence": 0,
                        "excluded": 1,
                        "requires_semantic_review": 3,
                    },
                },
            )
            write_json(
                phase_5c_path,
                {
                    "queue": {"reviewed_samples": 1},
                    "adjudication": {
                        "decision_counts": {
                            "high_confidence": 1,
                            "excluded": 0,
                            "uncertain": 0,
                        }
                    },
                    "updated_phase_5_population": {
                        "combined_high_confidence": 1,
                        "combined_dpo_v1_primary": 1,
                    },
                },
            )
            queue_hash = hashlib.sha256("extra\nmissing".encode()).hexdigest()
            write_json(
                annotations_path,
                {
                    "annotation_provenance": PROVENANCE,
                    "queue_sample_count": 2,
                    "queue_sample_ids_sha256": queue_hash,
                    "review_order": [
                        "missing_argument",
                        "extra_argument",
                        "nested_structure_mismatch",
                        "type_mismatch",
                    ],
                    "annotations": [
                        {
                            "sample_id": "missing",
                            "decision": "high_confidence",
                            "diagnostic_type": "missing_argument",
                            "semantic_reason": "The required value is explicit.",
                            "dpo_v1_primary": True,
                            "review_note": "Clear missing requirement.",
                        },
                        {
                            "sample_id": "extra",
                            "decision": "excluded",
                            "diagnostic_type": "extra_argument",
                            "semantic_reason": "The extra value is harmless.",
                            "dpo_v1_primary": False,
                            "review_note": "No preference direction.",
                        },
                    ],
                },
            )

            summary = adjudicate_non_wrong_values(
                candidate_cases_path=cases_path,
                candidate_summary_path=candidate_summary_path,
                filtering_summary_path=filtering_path,
                phase_5c_summary_path=phase_5c_path,
                annotations_path=annotations_path,
                output_cases_path=output_cases_path,
                output_summary_path=output_summary_path,
            )
            records = [
                json.loads(line)
                for line in output_cases_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            summary["adjudication"]["decision_counts"],
            {"high_confidence": 1, "excluded": 1, "uncertain": 0},
        )
        self.assertEqual(
            summary["final_5k_inventory"]["decision_counts"],
            {"high_confidence": 2, "excluded": 2, "uncertain": 0},
        )
        self.assertEqual(
            summary["decision_gate"]["threshold_band"],
            "below_120_consider_expanding_to_10k",
        )
        self.assertTrue(
            all(not record["preference_pair_materialized"] for record in records)
        )

    def test_primary_requires_high_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "candidate_cases.jsonl"
            annotations_path = root / "annotations.json"
            write_jsonl(cases_path, [candidate("a", "extra_argument")])
            write_json(
                annotations_path,
                {
                    "annotation_provenance": PROVENANCE,
                    "queue_sample_count": 1,
                    "queue_sample_ids_sha256": hashlib.sha256(
                        "a".encode()
                    ).hexdigest(),
                    "review_order": [
                        "missing_argument",
                        "extra_argument",
                        "nested_structure_mismatch",
                        "type_mismatch",
                    ],
                    "annotations": [
                        {
                            "sample_id": "a",
                            "decision": "excluded",
                            "diagnostic_type": "extra_argument",
                            "semantic_reason": "Harmless.",
                            "dpo_v1_primary": True,
                            "review_note": "Invalid primary assignment.",
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "only when high_confidence"):
                adjudicate_non_wrong_values(
                    candidate_cases_path=cases_path,
                    candidate_summary_path=root / "unused_candidate.json",
                    filtering_summary_path=root / "unused_filter.json",
                    phase_5c_summary_path=root / "unused_5c.json",
                    annotations_path=annotations_path,
                    output_cases_path=root / "out.jsonl",
                    output_summary_path=root / "out.json",
                )


if __name__ == "__main__":
    unittest.main()

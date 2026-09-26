"""Tests for Phase 5B train-side candidate filtering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.eval.analyze_dpo_mining import analyze_mining_results


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def tool(default: str | None = None) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "string", "description": "Test field."}
    if default is not None:
        field["default"] = default
    return {
        "type": "function",
        "function": {
            "name": "A",
            "description": "Test tool.",
            "parameters": {
                "type": "object",
                "properties": {"value": field},
            },
        },
    }


def gold(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "A", "arguments": arguments},
            }
        ],
    }


def make_records(
    sample_id: str,
    query: str,
    gold_arguments: dict[str, Any],
    pred_arguments: dict[str, Any],
    errors: list[str],
    *,
    default: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical = {
        "sample_id": sample_id,
        "source": "test",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": query},
        ],
        "tools": [tool(default)],
        "assistant": gold(gold_arguments),
        "metadata": {},
    }
    raw = json.dumps({"name": "A", "arguments": pred_arguments})
    prediction = {
        "sample_id": sample_id,
        "gold": canonical["assistant"],
        "raw_prediction": raw,
    }
    evaluation = {
        "sample_id": sample_id,
        "gold": canonical["assistant"],
        "raw_prediction": raw,
        "parsed_prediction": {
            "mode": "tool",
            "calls": [{"name": "A", "arguments": pred_arguments}],
        },
        "metrics": {
            "full_call_exact": False,
            "gold_call_kind": "single",
            "errors": errors,
        },
        "errors": errors,
    }
    return canonical, prediction, evaluation


class DpoMiningAnalysisTests(unittest.TestCase):
    def test_structural_and_preference_filters_remain_separate(self) -> None:
        records = [
            make_records(
                "reviewed_high",
                "Use alpha.",
                {"value": "alpha"},
                {"value": "beta"},
                ["argument_mismatch"],
            ),
            make_records(
                "default_equivalent",
                "Use the normal format.",
                {"value": "json"},
                {},
                ["argument_mismatch"],
                default="json",
            ),
            make_records(
                "pending",
                "Use the appropriate setting.",
                {"value": "gold"},
                {"value": "prediction"},
                ["argument_mismatch"],
            ),
            make_records(
                "mixed_error",
                "Use alpha.",
                {"value": "alpha"},
                {"value": "beta"},
                ["argument_mismatch", "wrong_tool"],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            predictions_path = root / "predictions.jsonl"
            evaluation_path = root / "evaluation.jsonl"
            summary_path = root / "summary.json"
            annotations_path = root / "annotations.json"
            write_jsonl(canonical_path, [record[0] for record in records])
            write_jsonl(predictions_path, [record[1] for record in records])
            write_jsonl(evaluation_path, [record[2] for record in records])
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol": {"frozen": True},
                        "inputs": {"partial_evaluation": False},
                        "metrics": {
                            "primary": {
                                "full_call_success_rate": {
                                    "value": 0.0,
                                    "numerator": 0,
                                    "denominator": 4,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            annotations_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "reviewed_high",
                            "decision": "high_confidence",
                            "semantic_bucket": "entity_or_literal_extraction",
                            "reason": "The requested literal is explicit.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            candidate_summary, filtering_summary = analyze_mining_results(
                summary_path=summary_path,
                evaluation_path=evaluation_path,
                predictions_path=predictions_path,
                canonical_path=canonical_path,
                annotations_path=annotations_path,
                candidate_summary_path=root / "candidate_summary.json",
                candidate_cases_path=root / "candidate_cases.jsonl",
                filtering_summary_path=root / "filtering_summary.json",
            )
            cases = [
                json.loads(line)
                for line in (root / "candidate_cases.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(
            candidate_summary["argument_diagnostics"][
                "clean_analyzable_candidate_samples"
            ],
            3,
        )
        self.assertEqual(
            filtering_summary["candidate_decision_counts"],
            {
                "high_confidence": 1,
                "excluded": 1,
                "requires_semantic_review": 1,
            },
        )
        self.assertEqual(len(cases), 3)
        self.assertFalse(any(case["preference_pair_materialized"] for case in cases))


if __name__ == "__main__":
    unittest.main()

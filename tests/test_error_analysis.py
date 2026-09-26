"""Tests for offline Base-to-SFT error-analysis alignment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.eval.analyze_errors import (
    analyze_failure_distribution,
    analyze_transition_errors,
    analyze_transitions,
    diagnose_argument_calls,
    diagnose_argument_sample,
    enrich_wrong_value_case,
    stratified_sample,
)


def evaluation(
    sample_id: str,
    correct: bool,
    call_type: str = "single",
    errors: list[str] | None = None,
) -> dict[str, Any]:
    evaluator_errors = [] if correct else (errors or ["argument_mismatch"])
    raw_prediction = f"prediction for {sample_id}"
    return {
        "sample_id": sample_id,
        "gold": {"content": None, "tool_calls": []},
        "raw_prediction": raw_prediction,
        "parsed_prediction": {"mode": "tool", "calls": []},
        "metrics": {
            "full_call_exact": correct,
            "gold_call_kind": call_type,
            "errors": evaluator_errors,
        },
        "errors": evaluator_errors,
    }


def prediction(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "gold": record["gold"],
        "raw_prediction": record["raw_prediction"],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def gold_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def gold(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"content": None, "tool_calls": list(calls)}


def parsed_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


def parsed(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"mode": "tool", "calls": list(calls)}


class ErrorAnalysisPhase1Tests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        run: str,
        split: str,
        records: list[dict[str, Any]],
    ) -> None:
        directory = root / run / split
        write_jsonl(directory / "evaluation.jsonl", records)
        write_jsonl(
            directory / "predictions.jsonl",
            [prediction(record) for record in records],
        )

    def test_four_quadrants_are_joined_and_summarized(self) -> None:
        base = [
            evaluation("still_correct", True, "single"),
            evaluation("fixed", False, "single"),
            evaluation("regressed", True, "multi"),
            evaluation("still_wrong", False, "multi"),
        ]
        sft = [
            evaluation("still_wrong", False, "multi"),
            evaluation("regressed", False, "multi"),
            evaluation("fixed", True, "single"),
            evaluation("still_correct", True, "single"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "base", "test", base)
            self._write_run(root, "sft_v1", "test", sft)
            output = root / "analysis" / "transition_summary.json"
            result = analyze_transitions(
                eval_root=root,
                output_path=output,
                splits=("test",),
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        counts = result["splits"]["test"]["counts"]
        self.assertEqual(
            counts["transitions"],
            {
                "still_correct": 1,
                "fixed_by_sft": 1,
                "regressed_by_sft": 1,
                "still_wrong": 1,
            },
        )
        self.assertEqual(counts["base_correct"], 2)
        self.assertEqual(counts["sft_correct"], 2)
        self.assertEqual(
            result["splits"]["test"]["by_call_type"]["single"]["counts"][
                "samples"
            ],
            2,
        )
        self.assertEqual(saved["phase"], "phase_1_transitions")

    def test_missing_sample_id_on_one_side_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "base", "test", [evaluation("s1", True)])
            self._write_run(root, "sft_v1", "test", [evaluation("s2", True)])
            with self.assertRaisesRegex(ValueError, "sample_id mismatch"):
                analyze_transitions(
                    eval_root=root,
                    output_path=root / "summary.json",
                    splits=("test",),
                )

    def test_prediction_evaluation_misalignment_is_rejected(self) -> None:
        record = evaluation("s1", True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "base", "test", [record])
            self._write_run(root, "sft_v1", "test", [record])
            predictions = root / "base" / "test" / "predictions.jsonl"
            stale = prediction(record)
            stale["raw_prediction"] = "stale prediction"
            write_jsonl(predictions, [stale])
            with self.assertRaisesRegex(ValueError, "raw prediction differ"):
                analyze_transitions(
                    eval_root=root,
                    output_path=root / "summary.json",
                    splits=("test",),
                )

    def test_sft_failure_distribution_preserves_multi_labels(self) -> None:
        base = [
            evaluation("correct", True),
            evaluation("single_multi_label", False),
            evaluation("multi_argument", False, "multi"),
            evaluation("multi_extra_label", False, "multi"),
        ]
        sft = [
            evaluation("correct", True),
            evaluation(
                "single_multi_label",
                False,
                errors=["argument_mismatch", "schema_invalid"],
            ),
            evaluation(
                "multi_argument",
                False,
                "multi",
                errors=["argument_mismatch"],
            ),
            evaluation(
                "multi_extra_label",
                False,
                "multi",
                errors=["future_evaluator_label"],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "base", "test", base)
            self._write_run(root, "sft_v1", "test", sft)
            result = analyze_failure_distribution(
                eval_root=root,
                output_path=root / "error_summary.json",
                splits=("test",),
            )

        summary = result["splits"]["test"]
        self.assertEqual(summary["failure_samples"], 3)
        self.assertEqual(summary["error_occurrences"], 4)
        self.assertEqual(summary["multi_label_failure_samples"], 1)
        self.assertEqual(summary["error_counts"]["argument_mismatch"], 2)
        self.assertEqual(summary["error_counts"]["schema_invalid"], 1)
        self.assertEqual(summary["error_counts"]["future_evaluator_label"], 1)
        self.assertEqual(
            summary["error_rates_among_failures"]["argument_mismatch"],
            2 / 3,
        )
        self.assertEqual(summary["by_call_type"]["single"]["failure_samples"], 1)
        self.assertEqual(summary["by_call_type"]["multi"]["failure_samples"], 2)

    def test_transition_error_breakdown_uses_the_correct_model_errors(self) -> None:
        base = [
            evaluation(
                "fixed_multi_label",
                False,
                errors=["argument_mismatch", "schema_invalid"],
            ),
            evaluation("fixed_custom_label", False, errors=["future_label"]),
            evaluation("regressed", True),
            evaluation("still_wrong", False, errors=["missing_tool"]),
        ]
        sft = [
            evaluation("fixed_multi_label", True),
            evaluation("fixed_custom_label", True),
            evaluation(
                "regressed",
                False,
                errors=["argument_mismatch", "wrong_tool"],
            ),
            evaluation("still_wrong", False, errors=["extra_tool"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "base", "test", base)
            self._write_run(root, "sft_v1", "test", sft)
            output = root / "transition_error_summary.json"
            result = analyze_transition_errors(
                eval_root=root,
                output_path=output,
                splits=("test",),
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        repaired = result["splits"]["test"]["repaired_base_errors"]
        regressed = result["splits"]["test"]["regression_sft_errors"]
        self.assertEqual(repaired["transition_samples"], 2)
        self.assertEqual(repaired["error_occurrences"], 3)
        self.assertEqual(repaired["multi_label_samples"], 1)
        self.assertEqual(repaired["error_counts"]["argument_mismatch"], 1)
        self.assertEqual(repaired["error_counts"]["schema_invalid"], 1)
        self.assertEqual(repaired["error_counts"]["future_label"], 1)
        self.assertEqual(repaired["error_counts"]["missing_tool"], 0)
        self.assertEqual(regressed["transition_samples"], 1)
        self.assertEqual(regressed["error_occurrences"], 2)
        self.assertEqual(regressed["error_counts"]["argument_mismatch"], 1)
        self.assertEqual(regressed["error_counts"]["wrong_tool"], 1)
        self.assertEqual(regressed["error_counts"]["extra_tool"], 0)
        self.assertTrue(saved["errors_are_non_mutually_exclusive"])


class ArgumentDiagnosticTests(unittest.TestCase):
    def test_missing_key(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"city": "Tokyo", "date": "2026-09-01"})),
            parsed(parsed_call("A", {"city": "Tokyo"})),
        )
        self.assertEqual(result["argument_diagnostics"], ["missing_argument"])
        self.assertEqual(result["diffs"][0]["path"], "date")
        self.assertFalse(result["diffs"][0]["pred_present"])

    def test_extra_key(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"city": "Tokyo"})),
            parsed(parsed_call("A", {"city": "Tokyo", "country": "Japan"})),
        )
        self.assertEqual(result["argument_diagnostics"], ["extra_argument"])
        self.assertEqual(result["diffs"][0]["path"], "country")
        self.assertFalse(result["diffs"][0]["gold_present"])

    def test_wrong_scalar_value(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"year": 2025})),
            parsed(parsed_call("A", {"year": 2024})),
        )
        self.assertEqual(result["argument_diagnostics"], ["wrong_value"])
        self.assertEqual(result["diffs"][0]["path"], "year")

    def test_type_mismatch_without_coercion(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"year": 2025})),
            parsed(parsed_call("A", {"year": "2025"})),
        )
        self.assertEqual(result["argument_diagnostics"], ["type_mismatch"])

    def test_integer_and_float_are_not_normalized(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"value": 1.0})),
            parsed(parsed_call("A", {"value": 1})),
        )
        self.assertEqual(result["argument_diagnostics"], ["type_mismatch"])

    def test_nested_dict_reports_precise_scalar_path(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"filters": {"price": {"max": 500}}})),
            parsed(parsed_call("A", {"filters": {"price": {"max": 600}}})),
        )
        self.assertEqual(result["argument_diagnostics"], ["wrong_value"])
        self.assertEqual(result["diffs"][0]["path"], "filters.price.max")

    def test_list_mismatch_stays_a_complex_structure_diagnostic(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"items": ["A", "B"]})),
            parsed(parsed_call("A", {"items": ["B", "A"]})),
        )
        self.assertEqual(
            result["argument_diagnostics"],
            ["nested_structure_mismatch"],
        )
        self.assertEqual(result["diffs"][0]["path"], "items")

    def test_multi_call_reorder_with_exact_calls_has_no_diagnostics(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"x": 1}), gold_call("B", {"y": 2})),
            parsed(parsed_call("B", {"y": 2}), parsed_call("A", {"x": 1})),
        )
        self.assertEqual(result["argument_diagnostics"], [])
        self.assertEqual(result["diffs"], [])
        self.assertEqual(result["matching"]["exact_call_pairs_removed"], 2)

    def test_multi_call_exact_plus_argument_mismatch_is_order_independent(self) -> None:
        result = diagnose_argument_calls(
            gold(gold_call("A", {"x": 1}), gold_call("B", {"y": 2})),
            parsed(parsed_call("B", {"y": 2}), parsed_call("A", {"x": 3})),
        )
        self.assertEqual(result["argument_diagnostics"], ["wrong_value"])
        self.assertEqual(len(result["diffs"]), 1)
        self.assertEqual(result["diffs"][0]["tool"], "A")
        self.assertEqual(result["diffs"][0]["path"], "x")
        self.assertEqual(result["matching"]["exact_call_pairs_removed"], 1)


class WrongValueAnalysisTests(unittest.TestCase):
    def test_wrong_value_case_is_enriched_from_canonical_context(self) -> None:
        canonical = {
            "sample_id": "s1",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Use economy class."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "book",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cabin": {
                                    "type": "string",
                                    "enum": ["economy", "business"],
                                }
                            },
                        },
                    },
                }
            ],
            "assistant": gold(gold_call("book", {"cabin": "economy"})),
        }
        case = {
            "sample_id": "s1",
            "split": "test_seen",
            "call_type": "single",
            "analysis_status": "analyzable",
            "argument_diagnostics": ["wrong_value"],
            "gold": canonical["assistant"],
            "diffs": [
                {
                    "tool": "book",
                    "path": "cabin",
                    "type": "wrong_value",
                    "gold": "economy",
                    "pred": "business",
                    "gold_call_index": 0,
                    "pred_call_index": 0,
                }
            ],
        }
        result = enrich_wrong_value_case(case, canonical)
        diff = result["wrong_value_diffs"][0]
        self.assertEqual(result["user_query"], "Use economy class.")
        self.assertEqual(diff["value_type"], "string")
        self.assertEqual(diff["schema_type"], "string")
        self.assertTrue(diff["enum_field"])

    def test_value_type_stratified_sampling_is_deterministic(self) -> None:
        cases = [
            {"sample_id": f"string_{index}", "sampling_value_type": "string"}
            for index in range(8)
        ] + [
            {
                "sample_id": f"number_{index}",
                "sampling_value_type": "integer_or_number",
            }
            for index in range(2)
        ]
        first = stratified_sample(cases, 5, seed=7, label="test")
        second = stratified_sample(cases, 5, seed=7, label="test")
        self.assertEqual(
            [case["sample_id"] for case in first],
            [case["sample_id"] for case in second],
        )
        self.assertEqual(len(first), 5)
        self.assertEqual(
            {case["sampling_value_type"] for case in first},
            {"string", "integer_or_number"},
        )
    def test_duplicate_tool_names_are_not_forcibly_paired(self) -> None:
        result = diagnose_argument_calls(
            gold(
                gold_call("search", {"q": "A"}),
                gold_call("search", {"q": "B"}),
            ),
            parsed(
                parsed_call("search", {"q": "C"}),
                parsed_call("search", {"q": "D"}),
            ),
        )
        self.assertEqual(result["analysis_status"], "ambiguous_or_confounded")
        self.assertEqual(result["argument_diagnostics"], ["ambiguous_call_pairing"])
        self.assertEqual(result["diffs"], [])

    def test_argument_mismatch_with_wrong_tool_is_only_partially_analyzed(self) -> None:
        row = {
            "sample_id": "mixed",
            "split": "test",
            "call_type": "multi",
            "gold": gold(
                gold_call("A", {"x": 1}),
                gold_call("B", {"y": 2}),
            ),
            "sft_prediction": parsed(
                parsed_call("A", {"x": 3}),
                parsed_call("C", {"z": 4}),
            ),
            "sft_errors": ["argument_mismatch", "wrong_tool", "missing_tool"],
        }
        result = diagnose_argument_sample(row)
        self.assertTrue(result["mixed_with_other_errors"])
        self.assertEqual(result["analysis_status"], "partially_analyzable")
        self.assertEqual(
            result["argument_diagnostics"],
            ["wrong_value", "partial_argument_analysis"],
        )
        self.assertEqual(len(result["diffs"]), 1)


if __name__ == "__main__":
    unittest.main()

"""Protocol tests for strict Step 4 parsing and metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.eval.evaluate import evaluate_files
from src.eval.metrics import aggregate_scores, canonical_arguments, score_sample
from src.eval.parser import parse_prediction


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant(*calls: dict[str, Any], content: str | None = None) -> dict[str, Any]:
    return {"content": content, "tool_calls": list(calls)}


def tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def tagged(name: str, arguments: dict[str, Any]) -> str:
    body = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{body}\n</tool_call>"


class ParserTests(unittest.TestCase):
    def test_strict_single_and_multi_tool_predictions(self) -> None:
        single = parse_prediction(tagged("A", {"x": 1}))
        multi = parse_prediction(
            tagged("A", {"x": 1}) + "\n" + tagged("B", {"y": 2})
        )
        self.assertEqual(single.mode, "tool")
        self.assertTrue(single.strict_format_valid)
        self.assertEqual([call.name for call in multi.calls], ["A", "B"])

    def test_text_and_mixed_are_distinct(self) -> None:
        text = parse_prediction("Please provide the project ID.")
        mixed = parse_prediction("I will call it.\n" + tagged("A", {}))
        self.assertEqual(text.mode, "text")
        self.assertEqual(mixed.mode, "mixed")
        self.assertFalse(mixed.strict_format_valid)
        self.assertEqual(mixed.calls[0].name, "A")

    def test_invalid_outputs_are_not_repaired(self) -> None:
        values = [
            "<tool_call>{'name':'A','arguments':{}}</tool_call>",
            '<tool_call>{"name":"A","arguments":"{}"}</tool_call>',
            '<tool_call>{"name":"A","arguments":{}}',
            '<tool_call>{"name":"A","arguments":{"x":NaN}}</tool_call>',
            "   ",
        ]
        self.assertTrue(all(parse_prediction(value).mode == "invalid" for value in values))

    def test_extra_call_fields_are_invalid(self) -> None:
        raw = '<tool_call>{"name":"A","arguments":{},"id":"1"}</tool_call>'
        self.assertEqual(parse_prediction(raw).mode, "invalid")


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = [
            tool("A", {"x": {"type": "integer"}}, ["x"]),
            tool("B", {"city": {"type": "string"}}, ["city"]),
        ]

    def test_argument_canonicalization_is_structural_only(self) -> None:
        self.assertEqual(
            canonical_arguments({"b": 2, "a": [1, 2]}),
            canonical_arguments({"a": [1, 2], "b": 2}),
        )
        self.assertNotEqual(
            canonical_arguments({"a": [1, 2]}),
            canonical_arguments({"a": [2, 1]}),
        )
        self.assertNotEqual(canonical_arguments({"x": 1}), canonical_arguments({"x": "1"}))

    def test_reordered_multi_calls_are_full_success(self) -> None:
        gold = assistant(call("A", {"x": 1}), call("B", {"city": "Beijing"}))
        parsed = parse_prediction(
            tagged("B", {"city": "Beijing"}) + "\n" + tagged("A", {"x": 1})
        )
        score = score_sample(gold, self.tools, parsed)
        self.assertTrue(score.full_call_exact)
        self.assertTrue(score.tool_name_exact)

    def test_call_multiplicity_is_preserved(self) -> None:
        gold = assistant(call("A", {"x": 1}), call("A", {"x": 1}))
        score = score_sample(gold, self.tools, parse_prediction(tagged("A", {"x": 1})))
        self.assertFalse(score.full_call_exact)
        self.assertFalse(score.tool_name_exact)
        self.assertIn("missing_tool", score.errors)

    def test_argument_em_is_conditioned_on_name_matches(self) -> None:
        gold = assistant(
            call("A", {"x": 1}),
            call("A", {"x": 2}),
            call("B", {"city": "Beijing"}),
        )
        parsed = parse_prediction(
            tagged("A", {"x": 2})
            + tagged("A", {"x": 5})
            + tagged("B", {"city": "Beijing"})
        )
        score = score_sample(gold, self.tools, parsed)
        self.assertEqual(score.argument_exact_matches, 2)
        self.assertEqual(score.name_matched_calls, 3)
        self.assertFalse(score.argument_exact)

    def test_schema_valid_can_be_true_when_argument_value_is_wrong(self) -> None:
        gold = assistant(call("B", {"city": "Beijing"}))
        parsed = parse_prediction(tagged("B", {"city": "Shanghai"}))
        score = score_sample(gold, self.tools, parsed)
        self.assertTrue(score.schema_valid)
        self.assertFalse(score.full_call_exact)
        self.assertIn("argument_mismatch", score.errors)

    def test_text_gold_only_scores_no_tool_choice(self) -> None:
        gold = assistant(content="Please provide the ID.")
        score = score_sample(gold, self.tools, parse_prediction("I need the ID."))
        self.assertTrue(score.no_tool_correct)
        self.assertTrue(score.response_mode_correct)
        self.assertFalse(score.full_call_exact)

    def test_mixed_output_has_valid_schema_but_fails_full_success(self) -> None:
        gold = assistant(call("A", {"x": 1}))
        parsed = parse_prediction("Calling now.\n" + tagged("A", {"x": 1}))
        score = score_sample(gold, self.tools, parsed)
        self.assertTrue(score.schema_valid)
        self.assertFalse(score.format_valid)
        self.assertFalse(score.full_call_exact)
        self.assertIn("format_error", score.errors)

    def test_tool_name_f1_is_zero_when_no_calls_are_predicted(self) -> None:
        gold = assistant(call("A", {"x": 1}))
        score = score_sample(gold, self.tools, parse_prediction("No tool needed."))
        metric = aggregate_scores([score])["diagnostic"]["tool_name_f1"]
        self.assertEqual(metric["value"], 0.0)
        self.assertEqual(metric["precision"], 0.0)
        self.assertEqual(metric["recall"], 0.0)


class EvaluatorTests(unittest.TestCase):
    def test_offline_join_and_metric_tiers(self) -> None:
        canonical = {
            "sample_id": "s1",
            "source": "xlam",
            "messages": [],
            "tools": [tool("A", {"x": {"type": "integer"}}, ["x"])],
            "assistant": assistant(call("A", {"x": 1})),
            "metadata": {},
        }
        prediction = {
            "sample_id": "s1",
            "gold": canonical["assistant"],
            "raw_prediction": tagged("A", {"x": 1}),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            predictions_path = root / "predictions.jsonl"
            evaluation_path = root / "evaluation.jsonl"
            summary_path = root / "summary.json"
            canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
            predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

            summary = evaluate_files(
                predictions_path,
                canonical_path,
                evaluation_path,
                summary_path,
            )
            records = evaluation_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(records), 1)
        self.assertTrue(summary["protocol"]["frozen"])
        self.assertIsNone(summary["protocol"]["composite_score"])
        self.assertEqual(
            summary["metrics"]["primary"]["full_call_success_rate"]["value"],
            1.0,
        )

    def test_stale_prediction_gold_is_rejected(self) -> None:
        canonical = {
            "sample_id": "s1",
            "source": "xlam",
            "messages": [],
            "tools": [tool("A")],
            "assistant": assistant(call("A", {})),
            "metadata": {},
        }
        prediction = {
            "sample_id": "s1",
            "gold": assistant(call("B", {})),
            "raw_prediction": tagged("A", {}),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            predictions_path = root / "predictions.jsonl"
            canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
            predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match canonical"):
                evaluate_files(
                    predictions_path,
                    canonical_path,
                    root / "evaluation.jsonl",
                    root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()

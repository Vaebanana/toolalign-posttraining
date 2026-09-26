"""Offline orchestration for the frozen Step 4 tool-use evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from .metrics import PROTOCOL_VERSION, SampleScore, aggregate_scores, score_sample
    from .parser import parse_prediction
except ImportError:  # Support ``python src/eval/evaluate.py``.
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.eval.metrics import (
        PROTOCOL_VERSION,
        SampleScore,
        aggregate_scores,
        score_sample,
    )
    from src.eval.parser import parse_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "eval" / "base" / "dev_seen" / "predictions.jsonl"
)
DEFAULT_CANONICAL = PROJECT_ROOT / "data" / "processed" / "dev_seen.jsonl"


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: record must be an object")
            yield line_number, value


def load_canonical_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for line_number, sample in read_jsonl(path):
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: invalid sample_id")
        if sample_id in index:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
        index[sample_id] = sample
    return index


def evaluate_prediction(
    prediction: dict[str, Any],
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], SampleScore]:
    sample_id = canonical["sample_id"]
    if prediction.get("gold") != canonical.get("assistant"):
        raise ValueError(
            f"{sample_id}: prediction gold does not match canonical assistant"
        )
    raw_prediction = prediction.get("raw_prediction")
    if not isinstance(raw_prediction, str):
        raise TypeError(f"{sample_id}: raw_prediction must be a string")
    tools = canonical.get("tools")
    if not isinstance(tools, list):
        raise TypeError(f"{sample_id}: canonical tools must be an array")

    parsed = parse_prediction(raw_prediction)
    score = score_sample(canonical["assistant"], tools, parsed)
    record = {
        "sample_id": sample_id,
        "source": canonical.get("source"),
        "metadata": canonical.get("metadata"),
        "gold": canonical["assistant"],
        "raw_prediction": raw_prediction,
        "parsed_prediction": parsed.to_dict(),
        "metrics": score.to_dict(),
        "errors": list(score.errors),
    }
    return record, score


def _protocol_manifest() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "frozen": True,
        "primary_metric": "full_call_success_rate",
        "secondary_metric": "no_tool_accuracy",
        "diagnostic_metrics": [
            "response_mode_accuracy",
            "tool_format_valid_rate",
            "tool_name_exact_match",
            "tool_name_f1",
            "argument_exact_match",
            "schema_valid_rate",
        ],
        "composite_score": None,
        "rules": {
            "prediction_json": "strict_no_repair",
            "prediction_modes": ["tool", "text", "mixed", "invalid"],
            "argument_normalization": "object_key_order_and_whitespace_only",
            "array_order": "preserved",
            "multi_call_comparison": "multiset",
            "schema_source": "candidate_tools_from_same_canonical_sample",
            "text_semantics": "not_automatically_scored",
            "metric_changes": (
                "bug_fixes_require_recomputing_all_model_results"
            ),
        },
    }


def build_summary(
    scores: list[SampleScore],
    parse_modes: Counter[str],
    *,
    predictions_path: Path,
    canonical_path: Path,
    canonical_count: int,
) -> dict[str, Any]:
    kinds = Counter(score.gold_call_kind for score in scores)
    error_counts = Counter(error for score in scores for error in score.errors)
    breakdowns: dict[str, Any] = {}
    for kind in ("single", "multi"):
        subset = [score for score in scores if score.gold_call_kind == kind]
        breakdowns[kind] = {
            "samples": len(subset),
            "full_call_success_rate": aggregate_scores(subset)["primary"][
                "full_call_success_rate"
            ],
            "tool_name_exact_match": aggregate_scores(subset)["diagnostic"][
                "tool_name_exact_match"
            ],
            "argument_exact_match": aggregate_scores(subset)["diagnostic"][
                "argument_exact_match"
            ],
        }

    return {
        "protocol": _protocol_manifest(),
        "inputs": {
            "predictions": str(predictions_path),
            "canonical": str(canonical_path),
            "partial_evaluation": len(scores) != canonical_count,
        },
        "counts": {
            "evaluated_samples": len(scores),
            "canonical_samples": canonical_count,
            "tool_samples": kinds["single"] + kinds["multi"],
            "text_samples": kinds["text"],
            "single_call_samples": kinds["single"],
            "multi_call_samples": kinds["multi"],
            "parse_modes": {
                mode: parse_modes[mode]
                for mode in ("tool", "text", "mixed", "invalid")
            },
        },
        "metrics": aggregate_scores(scores),
        "breakdowns": breakdowns,
        "error_counts": dict(sorted(error_counts.items())),
    }


def evaluate_files(
    predictions_path: Path,
    canonical_path: Path,
    evaluation_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions not found: {predictions_path}")
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical dataset not found: {canonical_path}")
    canonical_index = load_canonical_index(canonical_path)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    scores: list[SampleScore] = []
    parse_modes: Counter[str] = Counter()
    with evaluation_path.open("w", encoding="utf-8", newline="\n") as output:
        for line_number, prediction in read_jsonl(predictions_path):
            sample_id = prediction.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"{predictions_path}:{line_number}: invalid sample_id"
                )
            if sample_id in seen:
                raise ValueError(
                    f"{predictions_path}:{line_number}: duplicate sample_id "
                    f"{sample_id!r}"
                )
            seen.add(sample_id)
            canonical = canonical_index.get(sample_id)
            if canonical is None:
                raise ValueError(
                    f"{predictions_path}:{line_number}: unknown sample_id "
                    f"{sample_id!r}"
                )
            record, score = evaluate_prediction(prediction, canonical)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            scores.append(score)
            parse_modes[record["parsed_prediction"]["mode"]] += 1

    if not scores:
        raise ValueError("predictions file is empty")
    summary = build_summary(
        scores,
        parse_modes,
        predictions_path=predictions_path,
        canonical_path=canonical_path,
        canonical_count=len(canonical_index),
    )
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved tool-use predictions without model inference."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Per-sample JSONL (default: beside predictions.jsonl).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Aggregate JSON (default: beside predictions.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    canonical_path = args.canonical.resolve()
    evaluation_path = (
        args.evaluation.resolve()
        if args.evaluation is not None
        else predictions_path.with_name("evaluation.jsonl")
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else predictions_path.with_name("summary.json")
    )
    summary = evaluate_files(
        predictions_path,
        canonical_path,
        evaluation_path,
        summary_path,
    )
    primary = summary["metrics"]["primary"]["full_call_success_rate"]
    secondary = summary["metrics"]["secondary"]["no_tool_accuracy"]
    print("Step 4 evaluation complete")
    print(f"samples: {summary['counts']['evaluated_samples']}")
    print(f"full_call_success_rate: {primary['value']}")
    print(f"no_tool_accuracy: {secondary['value']}")
    print(f"evaluation: {evaluation_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

"""Offline error analysis for existing Base and SFT evaluator outputs.

Phase 1 aligns saved prediction/evaluation records by ``sample_id`` and
summarizes transitions in frozen ``full_call_exact`` outcomes.  It never
parses, repairs, or re-scores a prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "eval"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "analysis" / "sft_v1" / "transition_summary.json"
)
DEFAULT_ERROR_OUTPUT = (
    PROJECT_ROOT / "outputs" / "analysis" / "sft_v1" / "error_summary.json"
)
DEFAULT_TRANSITION_ERROR_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "sft_v1"
    / "transition_error_summary.json"
)
DEFAULT_ARGUMENT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "sft_v1"
    / "argument_error_summary.json"
)
DEFAULT_ARGUMENT_CASES = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "sft_v1"
    / "argument_error_cases.jsonl"
)
DEFAULT_WRONG_VALUE_OUTPUT = (
    PROJECT_ROOT / "outputs" / "analysis" / "sft_v1" / "wrong_value_summary.json"
)
DEFAULT_WRONG_VALUE_SPOTCHECK = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "sft_v1"
    / "wrong_value_spotcheck.jsonl"
)
DEFAULT_WRONG_VALUE_ANNOTATIONS = (
    PROJECT_ROOT
    / "configs"
    / "analysis"
    / "sft_v1_wrong_value_spotcheck_annotations.json"
)
DEFAULT_CANONICAL_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_SPOTCHECK_PER_SPLIT = 40
DEFAULT_SPOTCHECK_SEED = 20260831
DEFAULT_SPLITS = ("test_seen", "test_unseen_tools")
TRANSITIONS = (
    "still_correct",
    "fixed_by_sft",
    "regressed_by_sft",
    "still_wrong",
)
CALL_TYPES = ("single", "multi", "text")
EVALUATOR_ERROR_LABELS = (
    "argument_mismatch",
    "wrong_tool",
    "missing_tool",
    "extra_tool",
    "schema_invalid",
    "format_error",
    "wrong_mode",
)
ARGUMENT_DIAGNOSTIC_LABELS = (
    "wrong_value",
    "missing_argument",
    "extra_argument",
    "type_mismatch",
    "nested_structure_mismatch",
    "ambiguous_call_pairing",
    "partial_argument_analysis",
)
ARGUMENT_ANALYSIS_STATUSES = (
    "analyzable",
    "partially_analyzable",
    "ambiguous_or_confounded",
)
SEMANTIC_BUCKETS = (
    "entity_or_literal_extraction",
    "value_transformation_or_normalization",
    "constraint_or_schema_option_selection",
    "cross_field_binding",
    "derived_or_reasoning_value",
    "ambiguous_or_gold_issue",
)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield object records from a UTF-8 JSONL file with useful errors."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: record must be an object")
            yield line_number, value


def load_index(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    """Load JSONL records into a unique, non-empty ``sample_id`` index."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    index: dict[str, dict[str, Any]] = {}
    for line_number, record in read_jsonl(path):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: invalid sample_id")
        if sample_id in index:
            raise ValueError(
                f"{path}:{line_number}: duplicate sample_id {sample_id!r}"
            )
        index[sample_id] = record
    if not index:
        raise ValueError(f"{label} is empty: {path}")
    return index


def _require_same_ids(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> None:
    left_only = sorted(left.keys() - right.keys())
    right_only = sorted(right.keys() - left.keys())
    if left_only or right_only:
        details: list[str] = []
        if left_only:
            details.append(f"only in {left_label}: {left_only[:5]}")
        if right_only:
            details.append(f"only in {right_label}: {right_only[:5]}")
        raise ValueError("sample_id mismatch; " + "; ".join(details))


def _validate_evaluation_record(
    record: dict[str, Any],
    *,
    label: str,
    sample_id: str,
) -> tuple[bool, list[str], str]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"{label} {sample_id}: metrics must be an object")
    full_call_exact = metrics.get("full_call_exact")
    if not isinstance(full_call_exact, bool):
        raise TypeError(
            f"{label} {sample_id}: metrics.full_call_exact must be boolean"
        )
    errors = record.get("errors")
    metric_errors = metrics.get("errors")
    if (
        not isinstance(errors, list)
        or not all(isinstance(error, str) for error in errors)
    ):
        raise TypeError(f"{label} {sample_id}: errors must be an array of strings")
    if errors != metric_errors:
        raise ValueError(f"{label} {sample_id}: top-level and metric errors differ")
    call_type = metrics.get("gold_call_kind")
    if call_type not in CALL_TYPES:
        raise ValueError(f"{label} {sample_id}: invalid gold_call_kind {call_type!r}")
    return full_call_exact, errors, call_type


def load_run(
    evaluation_path: Path,
    predictions_path: Path,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Join one run's saved evaluation and prediction records."""
    evaluations = load_index(evaluation_path, label=f"{label} evaluation")
    predictions = load_index(predictions_path, label=f"{label} predictions")
    _require_same_ids(
        evaluations,
        predictions,
        left_label=f"{label} evaluation",
        right_label=f"{label} predictions",
    )
    for sample_id, evaluation in evaluations.items():
        prediction = predictions[sample_id]
        if evaluation.get("gold") != prediction.get("gold"):
            raise ValueError(
                f"{label} {sample_id}: evaluation and prediction gold differ"
            )
        if evaluation.get("raw_prediction") != prediction.get("raw_prediction"):
            raise ValueError(
                f"{label} {sample_id}: evaluation and raw prediction differ"
            )
        _validate_evaluation_record(
            evaluation,
            label=label,
            sample_id=sample_id,
        )
    return evaluations


def _transition(base_correct: bool, sft_correct: bool) -> str:
    if base_correct and sft_correct:
        return "still_correct"
    if not base_correct and sft_correct:
        return "fixed_by_sft"
    if base_correct and not sft_correct:
        return "regressed_by_sft"
    return "still_wrong"


def align_split(
    *,
    split: str,
    base_evaluation_path: Path,
    base_predictions_path: Path,
    sft_evaluation_path: Path,
    sft_predictions_path: Path,
) -> list[dict[str, Any]]:
    """Create the sample-level Base-to-SFT alignment used by all phases."""
    base = load_run(
        base_evaluation_path,
        base_predictions_path,
        label=f"base/{split}",
    )
    sft = load_run(
        sft_evaluation_path,
        sft_predictions_path,
        label=f"sft_v1/{split}",
    )
    _require_same_ids(
        base,
        sft,
        left_label=f"base/{split}",
        right_label=f"sft_v1/{split}",
    )

    aligned: list[dict[str, Any]] = []
    for sample_id, base_record in base.items():
        sft_record = sft[sample_id]
        if base_record.get("gold") != sft_record.get("gold"):
            raise ValueError(f"{split} {sample_id}: Base and SFT gold differ")
        base_correct, base_errors, base_call_type = _validate_evaluation_record(
            base_record,
            label=f"base/{split}",
            sample_id=sample_id,
        )
        sft_correct, sft_errors, sft_call_type = _validate_evaluation_record(
            sft_record,
            label=f"sft_v1/{split}",
            sample_id=sample_id,
        )
        if base_call_type != sft_call_type:
            raise ValueError(f"{split} {sample_id}: Base and SFT call types differ")
        aligned.append(
            {
                "sample_id": sample_id,
                "split": split,
                "gold": base_record["gold"],
                "base_prediction": base_record.get("parsed_prediction"),
                "sft_prediction": sft_record.get("parsed_prediction"),
                "base_raw_prediction": base_record["raw_prediction"],
                "sft_raw_prediction": sft_record["raw_prediction"],
                "base_full_call_exact": base_correct,
                "sft_full_call_exact": sft_correct,
                "base_errors": base_errors,
                "sft_errors": sft_errors,
                "call_type": base_call_type,
                "transition": _transition(base_correct, sft_correct),
            }
        )
    return aligned


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one aligned population without changing evaluator semantics."""
    transitions = Counter(row["transition"] for row in rows)
    transition_counts = {name: transitions[name] for name in TRANSITIONS}
    samples = len(rows)
    base_correct = transitions["still_correct"] + transitions["regressed_by_sft"]
    base_wrong = transitions["fixed_by_sft"] + transitions["still_wrong"]
    sft_correct = transitions["still_correct"] + transitions["fixed_by_sft"]
    sft_wrong = transitions["regressed_by_sft"] + transitions["still_wrong"]
    return {
        "counts": {
            "samples": samples,
            "base_correct": base_correct,
            "base_wrong": base_wrong,
            "sft_correct": sft_correct,
            "sft_wrong": sft_wrong,
            "transitions": transition_counts,
        },
        "rates": {
            "base_full_call_success_rate": _rate(base_correct, samples),
            "sft_full_call_success_rate": _rate(sft_correct, samples),
            "net_improvement": _rate(sft_correct - base_correct, samples),
            "fix_rate_among_base_failures": _rate(
                transitions["fixed_by_sft"], base_wrong
            ),
            "regression_rate_among_base_successes": _rate(
                transitions["regressed_by_sft"], base_correct
            ),
        },
    }


def _ordered_error_labels(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    actual_labels = {
        error
        for row in rows
        for error in row["sft_errors"]
    }
    extra_labels = sorted(actual_labels - set(EVALUATOR_ERROR_LABELS))
    return EVALUATOR_ERROR_LABELS + tuple(extra_labels)


def _ordered_transition_error_labels(
    rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    actual_labels = {
        error
        for row in rows
        for error in (
            row["base_errors"]
            if row["transition"] == "fixed_by_sft"
            else row["sft_errors"]
            if row["transition"] == "regressed_by_sft"
            else []
        )
    }
    extra_labels = sorted(actual_labels - set(EVALUATOR_ERROR_LABELS))
    return EVALUATOR_ERROR_LABELS + tuple(extra_labels)


def _transition_error_counts(
    rows: list[dict[str, Any]],
    *,
    error_field: str,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    errors_by_sample = [row[error_field] for row in rows]
    counts = Counter(error for errors in errors_by_sample for error in errors)
    return {
        "transition_samples": len(rows),
        "error_occurrences": sum(counts.values()),
        "samples_without_error_label": sum(
            not errors for errors in errors_by_sample
        ),
        "multi_label_samples": sum(len(errors) > 1 for errors in errors_by_sample),
        "error_counts": {label: counts[label] for label in labels},
    }


def _failure_counts(
    failures: list[dict[str, Any]],
    labels: tuple[str, ...],
    *,
    include_rates: bool,
) -> dict[str, Any]:
    counts = Counter(
        error
        for row in failures
        for error in row["sft_errors"]
    )
    result: dict[str, Any] = {
        "failure_samples": len(failures),
        "error_occurrences": sum(counts.values()),
        "failures_without_error_label": sum(
            not row["sft_errors"] for row in failures
        ),
        "multi_label_failure_samples": sum(
            len(row["sft_errors"]) > 1 for row in failures
        ),
        "error_counts": {label: counts[label] for label in labels},
    }
    if include_rates:
        denominator = len(failures)
        result["error_rates_among_failures"] = {
            label: counts[label] / denominator if denominator else None
            for label in labels
        }
    return result


def summarize_sft_failures(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Count frozen evaluator labels on SFT full-call failures."""
    failures = [row for row in rows if not row["sft_full_call_exact"]]
    if labels is None:
        labels = _ordered_error_labels(failures)
    summary = _failure_counts(failures, labels, include_rates=True)
    summary["by_call_type"] = {
        call_type: _failure_counts(
            [row for row in failures if row["call_type"] == call_type],
            labels,
            include_rates=False,
        )
        for call_type in CALL_TYPES
        if any(row["call_type"] == call_type for row in failures)
    }
    return summary


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"unsupported non-JSON value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def diff_arguments(
    gold: dict[str, Any],
    prediction: dict[str, Any],
    *,
    tool: str,
    gold_call_index: int | None = None,
    pred_call_index: int | None = None,
) -> list[dict[str, Any]]:
    """Recursively compare argument objects without normalization or repair."""
    diffs: list[dict[str, Any]] = []

    def add_diff(
        diagnostic: str,
        path: str,
        gold_value: Any,
        pred_value: Any,
        *,
        gold_present: bool = True,
        pred_present: bool = True,
    ) -> None:
        diffs.append(
            {
                "tool": tool,
                "gold_call_index": gold_call_index,
                "pred_call_index": pred_call_index,
                "path": path,
                "type": diagnostic,
                "gold_present": gold_present,
                "pred_present": pred_present,
                "gold": gold_value,
                "pred": pred_value,
            }
        )

    def compare(gold_value: Any, pred_value: Any, path: str) -> None:
        gold_type = _json_type(gold_value)
        pred_type = _json_type(pred_value)
        if gold_type != pred_type:
            add_diff("type_mismatch", path, gold_value, pred_value)
            return
        if gold_type == "object":
            gold_object = gold_value
            pred_object = pred_value
            for key in sorted(gold_object.keys() - pred_object.keys()):
                add_diff(
                    "missing_argument",
                    _path(path, key),
                    gold_object[key],
                    None,
                    pred_present=False,
                )
            for key in sorted(pred_object.keys() - gold_object.keys()):
                add_diff(
                    "extra_argument",
                    _path(path, key),
                    None,
                    pred_object[key],
                    gold_present=False,
                )
            for key in sorted(gold_object.keys() & pred_object.keys()):
                compare(gold_object[key], pred_object[key], _path(path, key))
            return
        if gold_type == "array":
            if _canonical_json(gold_value) != _canonical_json(pred_value):
                add_diff(
                    "nested_structure_mismatch",
                    path,
                    gold_value,
                    pred_value,
                )
            return
        if gold_value != pred_value:
            add_diff("wrong_value", path, gold_value, pred_value)

    compare(gold, prediction, "")
    return diffs


def _canonical_call_key(call: dict[str, Any]) -> tuple[str, str]:
    return (call["name"], _canonical_json(call["arguments"]))


def _read_gold_calls(gold: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = gold.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise TypeError("gold.tool_calls must be an array")
    calls: list[dict[str, Any]] = []
    for index, wrapper in enumerate(raw_calls):
        function = wrapper.get("function") if isinstance(wrapper, dict) else None
        if not isinstance(function, dict):
            raise TypeError(f"gold.tool_calls[{index}].function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError(f"gold.tool_calls[{index}] has invalid name/arguments")
        calls.append({"name": name, "arguments": arguments, "index": index})
    return calls


def _read_predicted_calls(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = prediction.get("calls")
    if not isinstance(raw_calls, list):
        raise TypeError("parsed prediction calls must be an array")
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            raise TypeError(f"parsed prediction calls[{index}] must be an object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError(
                f"parsed prediction calls[{index}] has invalid name/arguments"
            )
        calls.append({"name": name, "arguments": arguments, "index": index})
    return calls


def diagnose_argument_calls(
    gold: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Order-independently align calls for structural argument diagnosis."""
    gold_calls = _read_gold_calls(gold)
    pred_calls = _read_predicted_calls(prediction)
    unmatched_pred = set(range(len(pred_calls)))
    unmatched_gold: list[int] = []
    exact_pairs: list[tuple[int, int]] = []

    for gold_index, gold_call in enumerate(gold_calls):
        gold_key = _canonical_call_key(gold_call)
        pred_index = next(
            (
                index
                for index in sorted(unmatched_pred)
                if _canonical_call_key(pred_calls[index]) == gold_key
            ),
            None,
        )
        if pred_index is None:
            unmatched_gold.append(gold_index)
        else:
            unmatched_pred.remove(pred_index)
            exact_pairs.append((gold_index, pred_index))

    gold_by_name: dict[str, list[int]] = defaultdict(list)
    pred_by_name: dict[str, list[int]] = defaultdict(list)
    for index in unmatched_gold:
        gold_by_name[gold_calls[index]["name"]].append(index)
    for index in sorted(unmatched_pred):
        pred_by_name[pred_calls[index]["name"]].append(index)

    diffs: list[dict[str, Any]] = []
    compared_pairs: list[tuple[int, int]] = []
    ambiguous_names: list[str] = []
    unmatched_gold_calls: list[int] = []
    unmatched_pred_calls: list[int] = []
    for name in sorted(gold_by_name.keys() | pred_by_name.keys()):
        gold_indices = gold_by_name[name]
        pred_indices = pred_by_name[name]
        if len(gold_indices) == 1 and len(pred_indices) == 1:
            gold_index = gold_indices[0]
            pred_index = pred_indices[0]
            compared_pairs.append((gold_index, pred_index))
            diffs.extend(
                diff_arguments(
                    gold_calls[gold_index]["arguments"],
                    pred_calls[pred_index]["arguments"],
                    tool=name,
                    gold_call_index=gold_calls[gold_index]["index"],
                    pred_call_index=pred_calls[pred_index]["index"],
                )
            )
        elif gold_indices and pred_indices:
            ambiguous_names.append(name)
        else:
            unmatched_gold_calls.extend(gold_indices)
            unmatched_pred_calls.extend(pred_indices)

    diagnostics = {diff["type"] for diff in diffs}
    if ambiguous_names:
        diagnostics.add("ambiguous_call_pairing")
    has_unmatched_tools = bool(unmatched_gold_calls or unmatched_pred_calls)
    if has_unmatched_tools:
        diagnostics.add("partial_argument_analysis")

    if ambiguous_names:
        status = "ambiguous_or_confounded"
    elif has_unmatched_tools:
        status = (
            "partially_analyzable"
            if compared_pairs
            else "ambiguous_or_confounded"
        )
    else:
        status = "analyzable"

    return {
        "analysis_status": status,
        "argument_diagnostics": [
            label for label in ARGUMENT_DIAGNOSTIC_LABELS if label in diagnostics
        ],
        "diffs": diffs,
        "matching": {
            "gold_call_count": len(gold_calls),
            "predicted_call_count": len(pred_calls),
            "exact_call_pairs_removed": len(exact_pairs),
            "argument_pairs_compared": len(compared_pairs),
            "ambiguous_tool_names": ambiguous_names,
            "unmatched_gold_calls": len(unmatched_gold_calls),
            "unmatched_predicted_calls": len(unmatched_pred_calls),
        },
    }


def diagnose_argument_sample(row: dict[str, Any]) -> dict[str, Any]:
    if "argument_mismatch" not in row["sft_errors"]:
        raise ValueError(f"{row['sample_id']}: not an argument_mismatch sample")
    co_occurring_errors = [
        error for error in row["sft_errors"] if error != "argument_mismatch"
    ]
    diagnosis = diagnose_argument_calls(row["gold"], row["sft_prediction"])
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "call_type": row["call_type"],
        "argument_mismatch": True,
        "mixed_with_other_errors": bool(co_occurring_errors),
        "co_occurring_errors": co_occurring_errors,
        "sft_errors": row["sft_errors"],
        "analysis_status": diagnosis["analysis_status"],
        "argument_diagnostics": diagnosis["argument_diagnostics"],
        "diffs": diagnosis["diffs"],
        "matching": diagnosis["matching"],
        "gold": row["gold"],
        "sft_prediction": row["sft_prediction"],
    }


def _summarize_argument_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(cases)
    diagnostic_counts = Counter(
        diagnostic
        for case in cases
        for diagnostic in case["argument_diagnostics"]
    )
    diff_counts = Counter(
        diff["type"]
        for case in cases
        for diff in case["diffs"]
    )
    co_occurrences = Counter(
        error
        for case in cases
        for error in case["co_occurring_errors"]
    )
    statuses = Counter(case["analysis_status"] for case in cases)
    clean_count = sum(not case["mixed_with_other_errors"] for case in cases)
    return {
        "argument_mismatch_samples": sample_count,
        "clean_argument_failures": clean_count,
        "mixed_argument_failures": sample_count - clean_count,
        "co_occurrence_counts": dict(sorted(co_occurrences.items())),
        "status_counts": {
            status: statuses[status] for status in ARGUMENT_ANALYSIS_STATUSES
        },
        "diagnostic_sample_counts": {
            label: diagnostic_counts[label]
            for label in ARGUMENT_DIAGNOSTIC_LABELS
        },
        "diagnostic_rates_among_argument_failures": {
            label: diagnostic_counts[label] / sample_count if sample_count else None
            for label in ARGUMENT_DIAGNOSTIC_LABELS
        },
        "diff_occurrence_counts": {
            label: diff_counts[label]
            for label in ARGUMENT_DIAGNOSTIC_LABELS
            if diff_counts[label]
        },
    }


def summarize_argument_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_argument_cases(cases)
    summary["by_call_type"] = {
        call_type: _summarize_argument_cases(
            [case for case in cases if case["call_type"] == call_type]
        )
        for call_type in CALL_TYPES
        if any(case["call_type"] == call_type for case in cases)
    }
    return summary


def _wrong_value_type(value: Any) -> str:
    value_type = _json_type(value)
    if value_type in {"integer", "number"}:
        return "integer_or_number"
    return value_type


def _tool_index(tools: Any, *, sample_id: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(tools, list):
        raise TypeError(f"{sample_id}: canonical tools must be an array")
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, wrapper in enumerate(tools):
        function = wrapper.get("function") if isinstance(wrapper, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise TypeError(f"{sample_id}: tools[{position}] has invalid name")
        index[name].append(wrapper)
    return dict(index)


def _schema_at_path(tool: dict[str, Any], path: str) -> dict[str, Any] | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    schema = function.get("parameters")
    if not isinstance(schema, dict):
        return None
    current = schema
    for part in path.split(".") if path else []:
        properties = current.get("properties")
        if not isinstance(properties, dict):
            return None
        child = properties.get(part)
        if not isinstance(child, dict):
            return None
        current = child
    return current


def _user_query(messages: Any, *, sample_id: str) -> str:
    if not isinstance(messages, list):
        raise TypeError(f"{sample_id}: canonical messages must be an array")
    contents = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not contents or not all(isinstance(content, str) for content in contents):
        raise ValueError(f"{sample_id}: canonical user query is missing or invalid")
    return "\n".join(contents)


def enrich_wrong_value_case(
    case: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    """Attach query and schema context to an existing Phase 3 case."""
    sample_id = case["sample_id"]
    if canonical.get("assistant") != case.get("gold"):
        raise ValueError(f"{sample_id}: Phase 3 gold differs from canonical data")
    tools = _tool_index(canonical.get("tools"), sample_id=sample_id)
    wrong_diffs = [diff for diff in case["diffs"] if diff["type"] == "wrong_value"]
    if not wrong_diffs:
        raise ValueError(f"{sample_id}: wrong_value case has no wrong_value diff")

    enriched_diffs: list[dict[str, Any]] = []
    relevant_tool_names: list[str] = []
    for diff in wrong_diffs:
        tool_name = diff["tool"]
        tool_options = tools.get(tool_name)
        if tool_options is None:
            raise ValueError(f"{sample_id}: tool {tool_name!r} absent from canonical data")
        if tool_name not in relevant_tool_names:
            relevant_tool_names.append(tool_name)
        schema_options = [
            _schema_at_path(tool, diff["path"])
            for tool in tool_options
        ]
        unique_schemas = {
            _canonical_json(schema)
            for schema in schema_options
            if schema is not None
        }
        schema = (
            next((option for option in schema_options if option is not None), None)
            if len(unique_schemas) == 1
            else None
        )
        schema_type = schema.get("type") if schema is not None else None
        if isinstance(schema_type, list):
            schema_type = list(schema_type)
        elif not isinstance(schema_type, str):
            schema_type = None
        enriched_diffs.append(
            {
                "tool_name": tool_name,
                "argument_path": diff["path"],
                "gold_value": diff["gold"],
                "pred_value": diff["pred"],
                "value_type": _wrong_value_type(diff["gold"]),
                "schema_type": schema_type,
                "enum_field": bool(schema and isinstance(schema.get("enum"), list)),
                "schema_ambiguous": len(unique_schemas) > 1,
                "schema": schema,
                "gold_call_index": diff["gold_call_index"],
                "pred_call_index": diff["pred_call_index"],
            }
        )
    value_types = sorted({diff["value_type"] for diff in enriched_diffs})
    return {
        "sample_id": sample_id,
        "split": case["split"],
        "call_type": case["call_type"],
        "analysis_status": case["analysis_status"],
        "user_query": _user_query(canonical.get("messages"), sample_id=sample_id),
        "wrong_value_diffs": enriched_diffs,
        "sampling_value_type": value_types[0] if len(value_types) == 1 else "mixed",
        "relevant_tools": [
            tool
            for name in relevant_tool_names
            for tool in tools[name]
        ],
    }


def _stable_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def stratified_sample(
    cases: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    """Deterministically sample across the compact value-type strata."""
    if count < 0 or count > len(cases):
        raise ValueError(f"invalid sample count {count} for population {len(cases)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["sampling_value_type"]].append(case)
    for stratum, group in groups.items():
        _stable_rng(seed, f"{label}:{stratum}").shuffle(group)

    selected: dict[str, int] = {stratum: 0 for stratum in groups}
    if count >= len(groups):
        for stratum in groups:
            selected[stratum] = 1
    while sum(selected.values()) < count:
        available = [
            stratum
            for stratum, group in groups.items()
            if selected[stratum] < len(group)
        ]
        if not available:
            raise RuntimeError("stratified sampler exhausted before reaching target")
        stratum = min(
            available,
            key=lambda name: (
                selected[name] / len(groups[name]),
                name,
            ),
        )
        selected[stratum] += 1

    sample = [
        case
        for stratum in sorted(groups)
        for case in groups[stratum][: selected[stratum]]
    ]
    _stable_rng(seed, f"{label}:final").shuffle(sample)
    return sample


def _load_spotcheck_annotations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path}: annotations must be an array")
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for index, annotation in enumerate(value):
        if not isinstance(annotation, dict):
            raise TypeError(f"{path}: annotation {index} must be an object")
        split = annotation.get("split")
        sample_id = annotation.get("sample_id")
        bucket = annotation.get("semantic_bucket")
        if not isinstance(split, str) or not isinstance(sample_id, str):
            raise TypeError(f"{path}: annotation {index} has invalid identity")
        if bucket not in SEMANTIC_BUCKETS:
            raise ValueError(f"{path}: annotation {index} has invalid bucket {bucket!r}")
        for field in (
            "gold_explicit_in_query",
            "requires_reasoning",
            "dpo_candidate",
        ):
            if not isinstance(annotation.get(field), bool):
                raise TypeError(f"{path}: annotation {index}.{field} must be boolean")
        key = (split, sample_id)
        if key in annotations:
            raise ValueError(f"{path}: duplicate annotation for {key}")
        annotations[key] = annotation
    return annotations


def _automatic_wrong_value_distribution(cases: list[dict[str, Any]]) -> dict[str, Any]:
    value_samples: Counter[str] = Counter()
    value_diffs: Counter[str] = Counter()
    schema_samples: Counter[str] = Counter()
    schema_diffs: Counter[str] = Counter()
    enum_samples = 0
    enum_diffs = 0
    schema_ambiguous_samples = 0
    schema_ambiguous_diffs = 0
    for case in cases:
        case_value_types = {diff["value_type"] for diff in case["wrong_value_diffs"]}
        case_schema_types = {
            str(diff["schema_type"] or "unknown")
            for diff in case["wrong_value_diffs"]
        }
        value_samples.update(case_value_types)
        schema_samples.update(case_schema_types)
        if any(diff["enum_field"] for diff in case["wrong_value_diffs"]):
            enum_samples += 1
        if any(diff["schema_ambiguous"] for diff in case["wrong_value_diffs"]):
            schema_ambiguous_samples += 1
        for diff in case["wrong_value_diffs"]:
            value_diffs[diff["value_type"]] += 1
            schema_diffs[str(diff["schema_type"] or "unknown")] += 1
            enum_diffs += diff["enum_field"]
            schema_ambiguous_diffs += diff["schema_ambiguous"]
    return {
        "samples": len(cases),
        "wrong_value_diff_occurrences": sum(value_diffs.values()),
        "value_type_sample_counts": dict(sorted(value_samples.items())),
        "value_type_diff_counts": dict(sorted(value_diffs.items())),
        "schema_type_sample_counts": dict(sorted(schema_samples.items())),
        "schema_type_diff_counts": dict(sorted(schema_diffs.items())),
        "enum_field_sample_count": enum_samples,
        "enum_field_diff_count": enum_diffs,
        "schema_ambiguous_sample_count": schema_ambiguous_samples,
        "schema_ambiguous_diff_count": schema_ambiguous_diffs,
    }


def _semantic_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = [record for record in records if record["semantic_bucket"] is not None]
    bucket_counts = Counter(record["semantic_bucket"] for record in annotated)
    dpo_by_bucket = Counter(
        record["semantic_bucket"]
        for record in annotated
        if record["dpo_candidate"] is True
    )
    denominator = len(annotated)
    return {
        "annotated_cases": denominator,
        "semantic_bucket_counts": {
            bucket: bucket_counts[bucket] for bucket in SEMANTIC_BUCKETS
        },
        "semantic_bucket_rates": {
            bucket: bucket_counts[bucket] / denominator if denominator else None
            for bucket in SEMANTIC_BUCKETS
        },
        "dpo_candidate_counts_by_bucket": {
            bucket: dpo_by_bucket[bucket] for bucket in SEMANTIC_BUCKETS
        },
        "gold_explicit_in_query": sum(
            record["gold_explicit_in_query"] is True for record in annotated
        ),
        "requires_reasoning": sum(
            record["requires_reasoning"] is True for record in annotated
        ),
        "dpo_candidates": sum(
            record["dpo_candidate"] is True for record in annotated
        ),
    }


def _split_paths(eval_root: Path, run: str, split: str) -> tuple[Path, Path]:
    directory = eval_root / run / split
    return directory / "evaluation.jsonl", directory / "predictions.jsonl"


def analyze_transitions(
    *,
    eval_root: Path,
    output_path: Path,
    base_run: str = "base",
    sft_run: str = "sft_v1",
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Run Phase 1 and write ``transition_summary.json``."""
    if not splits or any(not split for split in splits):
        raise ValueError("at least one non-empty split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in splits:
        base_evaluation, base_predictions = _split_paths(eval_root, base_run, split)
        sft_evaluation, sft_predictions = _split_paths(eval_root, sft_run, split)
        rows = align_split(
            split=split,
            base_evaluation_path=base_evaluation,
            base_predictions_path=base_predictions,
            sft_evaluation_path=sft_evaluation,
            sft_predictions_path=sft_predictions,
        )
        summary = summarize_rows(rows)
        summary["inputs"] = {
            "base_evaluation": str(base_evaluation.resolve()),
            "base_predictions": str(base_predictions.resolve()),
            "sft_evaluation": str(sft_evaluation.resolve()),
            "sft_predictions": str(sft_predictions.resolve()),
        }
        summary["by_call_type"] = {
            call_type: summarize_rows(
                [row for row in rows if row["call_type"] == call_type]
            )
            for call_type in CALL_TYPES
            if any(row["call_type"] == call_type for row in rows)
        }
        split_summaries[split] = summary
        all_rows.extend(rows)

    result = {
        "analysis": "sft_v1_error_analysis",
        "phase": "phase_1_transitions",
        "frozen_evaluator_outputs_only": True,
        "models": {"base": base_run, "sft": sft_run},
        "transition_definitions": {
            "still_correct": "Base correct -> SFT correct",
            "fixed_by_sft": "Base wrong -> SFT correct",
            "regressed_by_sft": "Base correct -> SFT wrong",
            "still_wrong": "Base wrong -> SFT wrong",
        },
        "splits": split_summaries,
        "overall": summarize_rows(all_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def analyze_failure_distribution(
    *,
    eval_root: Path,
    output_path: Path,
    base_run: str = "base",
    sft_run: str = "sft_v1",
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Run Phase 2 and write the SFT failure error distribution."""
    if not splits or any(not split for split in splits):
        raise ValueError("at least one non-empty split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        base_evaluation, base_predictions = _split_paths(eval_root, base_run, split)
        sft_evaluation, sft_predictions = _split_paths(eval_root, sft_run, split)
        rows_by_split[split] = align_split(
            split=split,
            base_evaluation_path=base_evaluation,
            base_predictions_path=base_predictions,
            sft_evaluation_path=sft_evaluation,
            sft_predictions_path=sft_predictions,
        )

    all_rows = [
        row
        for split in splits
        for row in rows_by_split[split]
    ]
    all_failures = [row for row in all_rows if not row["sft_full_call_exact"]]
    labels = _ordered_error_labels(all_failures)
    result = {
        "analysis": "sft_v1_error_analysis",
        "phase": "phase_2_sft_failure_distribution",
        "frozen_evaluator_outputs_only": True,
        "errors_are_non_mutually_exclusive": True,
        "models": {"base": base_run, "sft": sft_run},
        "splits": {
            split: summarize_sft_failures(
                rows_by_split[split],
                labels=labels,
            )
            for split in splits
        },
        "overall": summarize_sft_failures(all_rows, labels=labels),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def analyze_transition_errors(
    *,
    eval_root: Path,
    output_path: Path,
    base_run: str = "base",
    sft_run: str = "sft_v1",
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Count frozen error labels conditioned on repairs and regressions."""
    if not splits or any(not split for split in splits):
        raise ValueError("at least one non-empty split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        base_evaluation, base_predictions = _split_paths(eval_root, base_run, split)
        sft_evaluation, sft_predictions = _split_paths(eval_root, sft_run, split)
        rows_by_split[split] = align_split(
            split=split,
            base_evaluation_path=base_evaluation,
            base_predictions_path=base_predictions,
            sft_evaluation_path=sft_evaluation,
            sft_predictions_path=sft_predictions,
        )

    all_rows = [row for split in splits for row in rows_by_split[split]]
    labels = _ordered_transition_error_labels(all_rows)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        repaired = [row for row in rows if row["transition"] == "fixed_by_sft"]
        regressed = [
            row for row in rows if row["transition"] == "regressed_by_sft"
        ]
        return {
            "repaired_base_errors": _transition_error_counts(
                repaired,
                error_field="base_errors",
                labels=labels,
            ),
            "regression_sft_errors": _transition_error_counts(
                regressed,
                error_field="sft_errors",
                labels=labels,
            ),
        }

    result = {
        "analysis": "sft_v1_error_analysis",
        "phase": "transition_conditioned_error_breakdown",
        "frozen_evaluator_outputs_only": True,
        "errors_are_non_mutually_exclusive": True,
        "models": {"base": base_run, "sft": sft_run},
        "breakdown_definitions": {
            "repaired_base_errors": {
                "transition": "Base wrong -> SFT correct",
                "error_source": "Base frozen evaluation errors",
                "question": "What did SFT repair?",
            },
            "regression_sft_errors": {
                "transition": "Base correct -> SFT wrong",
                "error_source": "SFT frozen evaluation errors",
                "question": "What did SFT break?",
            },
        },
        "splits": {split: summarize(rows_by_split[split]) for split in splits},
        "overall": summarize(all_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def analyze_argument_errors(
    *,
    eval_root: Path,
    summary_path: Path,
    cases_path: Path,
    base_run: str = "base",
    sft_run: str = "sft_v1",
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Run Phase 3 structural diagnostics on evaluator argument failures."""
    if not splits or any(not split for split in splits):
        raise ValueError("at least one non-empty split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")

    cases_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        base_evaluation, base_predictions = _split_paths(eval_root, base_run, split)
        sft_evaluation, sft_predictions = _split_paths(eval_root, sft_run, split)
        rows = align_split(
            split=split,
            base_evaluation_path=base_evaluation,
            base_predictions_path=base_predictions,
            sft_evaluation_path=sft_evaluation,
            sft_predictions_path=sft_predictions,
        )
        cases_by_split[split] = [
            diagnose_argument_sample(row)
            for row in rows
            if "argument_mismatch" in row["sft_errors"]
        ]

    all_cases = [
        case
        for split in splits
        for case in cases_by_split[split]
    ]
    result = {
        "analysis": "sft_v1_error_analysis",
        "phase": "phase_3_argument_diagnostics",
        "frozen_evaluator_outputs_only": True,
        "diagnostics_are_non_mutually_exclusive": True,
        "diagnostic_only_no_metric_changes": True,
        "semantic_normalization_or_repair": False,
        "test_cases_must_not_be_used_for_training": True,
        "matching_protocol": [
            "remove exact calls using frozen canonical JSON representation",
            "match remaining calls by unique tool name without using position",
            "do not force-pair ambiguous duplicate tool names",
        ],
        "diagnostic_definitions": {
            "wrong_value": "same concrete JSON type, unequal scalar value",
            "missing_argument": "gold object key absent from prediction",
            "extra_argument": "prediction object key absent from gold",
            "type_mismatch": (
                "different concrete JSON representation types; integer and "
                "floating-point representations remain distinct"
            ),
            "nested_structure_mismatch": (
                "strict unequal array or other intentionally unsplit structure"
            ),
            "ambiguous_call_pairing": (
                "remaining duplicate tool names cannot be uniquely paired"
            ),
            "partial_argument_analysis": (
                "safe same-tool argument pairs coexist with unmatched tools"
            ),
        },
        "models": {"base": base_run, "sft": sft_run},
        "cases": str(cases_path.resolve()),
        "splits": {
            split: summarize_argument_cases(cases_by_split[split])
            for split in splits
        },
        "overall": summarize_argument_cases(all_cases),
    }

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in all_cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def analyze_wrong_values(
    *,
    argument_cases_path: Path,
    canonical_root: Path,
    summary_path: Path,
    spotcheck_path: Path,
    annotations_path: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    spotcheck_per_split: int = DEFAULT_SPOTCHECK_PER_SPLIT,
    seed: int = DEFAULT_SPOTCHECK_SEED,
) -> dict[str, Any]:
    """Run Phase 4 automatic profiling and prepare the semantic spot check."""
    phase3_cases = [record for _, record in read_jsonl(argument_cases_path)]
    wrong_cases_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in splits
    }
    for case in phase3_cases:
        split = case.get("split")
        if split in wrong_cases_by_split and "wrong_value" in case.get(
            "argument_diagnostics", []
        ):
            wrong_cases_by_split[split].append(case)

    enriched_by_split: dict[str, list[dict[str, Any]]] = {}
    eligible_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        canonical_path = canonical_root / f"{split}.jsonl"
        canonical = load_index(canonical_path, label=f"canonical {split}")
        enriched = []
        for case in wrong_cases_by_split[split]:
            sample_id = case["sample_id"]
            sample = canonical.get(sample_id)
            if sample is None:
                raise ValueError(f"{split} {sample_id}: absent from canonical data")
            enriched.append(enrich_wrong_value_case(case, sample))
        enriched_by_split[split] = enriched
        eligible_by_split[split] = [
            case for case in enriched if case["analysis_status"] == "analyzable"
        ]

    selected: list[dict[str, Any]] = []
    for split in splits:
        eligible = eligible_by_split[split]
        by_call_type = {
            call_type: [case for case in eligible if case["call_type"] == call_type]
            for call_type in ("single", "multi")
        }
        target = min(spotcheck_per_split, len(eligible))
        quotas = {"single": target // 2, "multi": target - target // 2}
        for call_type in ("single", "multi"):
            shortfall = max(0, quotas[call_type] - len(by_call_type[call_type]))
            quotas[call_type] -= shortfall
            other = "multi" if call_type == "single" else "single"
            quotas[other] = min(
                len(by_call_type[other]),
                quotas[other] + shortfall,
            )
        if sum(quotas.values()) != target:
            raise ValueError(f"{split}: unable to fill spot-check target {target}")
        for call_type in ("single", "multi"):
            selected.extend(
                stratified_sample(
                    by_call_type[call_type],
                    quotas[call_type],
                    seed=seed,
                    label=f"{split}:{call_type}",
                )
            )

    annotations = _load_spotcheck_annotations(annotations_path)
    selected_keys = {(case["split"], case["sample_id"]) for case in selected}
    stale_annotations = sorted(annotations.keys() - selected_keys)
    if stale_annotations:
        raise ValueError(
            f"{annotations_path}: annotations not in deterministic sample: "
            f"{stale_annotations[:5]}"
        )

    spotcheck_records: list[dict[str, Any]] = []
    for case in selected:
        key = (case["split"], case["sample_id"])
        annotation = annotations.get(key)
        record = dict(case)
        record.update(
            {
                "semantic_bucket": (
                    annotation["semantic_bucket"] if annotation else None
                ),
                "gold_explicit_in_query": (
                    annotation["gold_explicit_in_query"] if annotation else None
                ),
                "requires_reasoning": (
                    annotation["requires_reasoning"] if annotation else None
                ),
                "dpo_candidate": (
                    annotation["dpo_candidate"] if annotation else None
                ),
                "review_note": annotation.get("review_note") if annotation else None,
                "review_status": "reviewed" if annotation else "pending",
            }
        )
        spotcheck_records.append(record)

    all_population = [
        case
        for split in splits
        for case in enriched_by_split[split]
    ]
    population_summary: dict[str, Any] = {}
    automatic_summary: dict[str, Any] = {}
    semantic_by_split: dict[str, Any] = {}
    for split in splits:
        population = enriched_by_split[split]
        eligible = eligible_by_split[split]
        split_spotcheck = [
            record for record in spotcheck_records if record["split"] == split
        ]
        statuses = Counter(case["analysis_status"] for case in population)
        population_summary[split] = {
            "wrong_value_samples": len(population),
            "analyzable_samples": len(eligible),
            "excluded_partially_analyzable": statuses["partially_analyzable"],
            "excluded_ambiguous_or_confounded": statuses[
                "ambiguous_or_confounded"
            ],
        }
        automatic_summary[split] = {
            "overall": _automatic_wrong_value_distribution(population),
            "by_call_type": {
                call_type: _automatic_wrong_value_distribution(
                    [case for case in population if case["call_type"] == call_type]
                )
                for call_type in ("single", "multi")
            },
        }
        semantic_by_split[split] = _semantic_distribution(split_spotcheck)

    semantic_by_call_type = {
        call_type: _semantic_distribution(
            [
                record
                for record in spotcheck_records
                if record["call_type"] == call_type
            ]
        )
        for call_type in ("single", "multi")
    }
    overall_semantic = _semantic_distribution(spotcheck_records)
    ranked_buckets = sorted(
        SEMANTIC_BUCKETS,
        key=lambda bucket: (
            -overall_semantic["semantic_bucket_counts"][bucket],
            bucket,
        ),
    )
    dpo_targets = sorted(
        (
            bucket
            for bucket in SEMANTIC_BUCKETS
            if bucket != "ambiguous_or_gold_issue"
            and overall_semantic["dpo_candidate_counts_by_bucket"][bucket]
        ),
        key=lambda bucket: (
            -overall_semantic["dpo_candidate_counts_by_bucket"][bucket],
            bucket,
        ),
    )
    primary_dpo_targets = dpo_targets[:3]
    primary_dpo_target_count = sum(
        overall_semantic["dpo_candidate_counts_by_bucket"][bucket]
        for bucket in primary_dpo_targets
    )
    selection_counts = {
        split: {
            "samples": sum(record["split"] == split for record in spotcheck_records),
            "single": sum(
                record["split"] == split and record["call_type"] == "single"
                for record in spotcheck_records
            ),
            "multi": sum(
                record["split"] == split and record["call_type"] == "multi"
                for record in spotcheck_records
            ),
            "by_value_type": dict(
                sorted(
                    Counter(
                        record["sampling_value_type"]
                        for record in spotcheck_records
                        if record["split"] == split
                    ).items()
                )
            ),
        }
        for split in splits
    }
    result = {
        "analysis": "sft_v1_error_analysis",
        "phase": "phase_4_wrong_value_semantic_analysis",
        "source": str(argument_cases_path.resolve()),
        "frozen_evaluator_unchanged": True,
        "semantic_repair": False,
        "test_cases_must_not_be_used_for_training": True,
        "semantic_labels_are_spotcheck_annotations_not_ground_truth": True,
        "annotation_provenance": "codex_assisted_case_by_case_spotcheck",
        "population": {
            **population_summary,
            "overall_wrong_value_samples": len(all_population),
        },
        "automatic_value_and_schema_distribution": automatic_summary,
        "spotcheck": {
            "seed": seed,
            "requested_per_split": spotcheck_per_split,
            "selection_counts": selection_counts,
            "total_samples": len(spotcheck_records),
            "reviewed_samples": sum(
                record["review_status"] == "reviewed"
                for record in spotcheck_records
            ),
            "pending_samples": sum(
                record["review_status"] == "pending"
                for record in spotcheck_records
            ),
            "annotations_source": str(annotations_path.resolve()),
        },
        "semantic_bucket_distribution": {
            "overall": overall_semantic,
            "by_split": semantic_by_split,
            "by_call_type": semantic_by_call_type,
        },
        "seen_vs_unseen": {
            split: semantic_by_split[split] for split in splits
        },
        "seen_vs_unseen_interpretation": (
            "The stratified spot check shows similar leading error structure; "
            "it does not establish equality of the full population distributions."
        ),
        "spotcheck_conclusions": {
            "top_three_semantic_buckets": [
                {
                    "bucket": bucket,
                    "count": overall_semantic["semantic_bucket_counts"][bucket],
                    "rate": overall_semantic["semantic_bucket_rates"][bucket],
                }
                for bucket in ranked_buckets[:3]
            ],
            "all_high_confidence_dpo_candidates": overall_semantic[
                "dpo_candidates"
            ],
            "dpo_v1_primary_target_cases": primary_dpo_target_count,
            "dpo_v1_primary_target_buckets": [
                {
                    "bucket": bucket,
                    "spotcheck_dpo_candidates": overall_semantic[
                        "dpo_candidate_counts_by_bucket"
                    ][bucket],
                }
                for bucket in primary_dpo_targets
            ],
            "residual_wrong_value_pattern": {
                "gold_explicit_in_query": overall_semantic[
                    "gold_explicit_in_query"
                ],
                "requires_reasoning": overall_semantic["requires_reasoning"],
                "interpretation": (
                    "Most reviewed residual errors involve mapping query semantics "
                    "through tool descriptions or schemas to an exact argument "
                    "value, rather than direct literal copying."
                ),
            },
            "preference_quality_filtering_rule": [
                "strict evaluator mismatch is not sufficient for a preference pair",
                "require a high-confidence chosen-over-rejected direction",
                "exclude semantically equivalent or underdetermined gold cases",
            ],
            "training_data_rule": (
                "regenerate analogous hard negatives from train; never train on "
                "these test cases"
            ),
        },
    }

    spotcheck_path.parent.mkdir(parents=True, exist_ok=True)
    with spotcheck_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in spotcheck_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: align existing Base/SFT evaluator outputs and summarize "
            "full-call transitions without re-scoring."
        )
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--error-output",
        type=Path,
        default=DEFAULT_ERROR_OUTPUT,
        help="Phase 2 SFT failure distribution JSON.",
    )
    parser.add_argument(
        "--transition-error-output",
        type=Path,
        default=DEFAULT_TRANSITION_ERROR_OUTPUT,
        help="Transition-conditioned repaired/regression error counts JSON.",
    )
    parser.add_argument(
        "--argument-output",
        type=Path,
        default=DEFAULT_ARGUMENT_OUTPUT,
        help="Phase 3 argument diagnostic summary JSON.",
    )
    parser.add_argument(
        "--argument-cases",
        type=Path,
        default=DEFAULT_ARGUMENT_CASES,
        help="Phase 3 sample-level argument diagnostic JSONL.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT,
        help="Canonical split directory used to enrich Phase 4 cases.",
    )
    parser.add_argument(
        "--wrong-value-output",
        type=Path,
        default=DEFAULT_WRONG_VALUE_OUTPUT,
        help="Phase 4 wrong-value summary JSON.",
    )
    parser.add_argument(
        "--wrong-value-spotcheck",
        type=Path,
        default=DEFAULT_WRONG_VALUE_SPOTCHECK,
        help="Phase 4 stratified semantic spot-check JSONL.",
    )
    parser.add_argument(
        "--wrong-value-annotations",
        type=Path,
        default=DEFAULT_WRONG_VALUE_ANNOTATIONS,
        help="Reviewed annotations merged into the Phase 4 spot check.",
    )
    parser.add_argument(
        "--spotcheck-per-split",
        type=int,
        default=DEFAULT_SPOTCHECK_PER_SPLIT,
    )
    parser.add_argument(
        "--spotcheck-seed",
        type=int,
        default=DEFAULT_SPOTCHECK_SEED,
    )
    parser.add_argument("--base-run", default="base")
    parser.add_argument("--sft-run", default="sft_v1")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    result = analyze_transitions(
        eval_root=args.eval_root.resolve(),
        output_path=output_path,
        base_run=args.base_run,
        sft_run=args.sft_run,
        splits=tuple(args.splits),
    )
    error_output_path = args.error_output.resolve()
    error_result = analyze_failure_distribution(
        eval_root=args.eval_root.resolve(),
        output_path=error_output_path,
        base_run=args.base_run,
        sft_run=args.sft_run,
        splits=tuple(args.splits),
    )
    transition_error_output_path = args.transition_error_output.resolve()
    transition_error_result = analyze_transition_errors(
        eval_root=args.eval_root.resolve(),
        output_path=transition_error_output_path,
        base_run=args.base_run,
        sft_run=args.sft_run,
        splits=tuple(args.splits),
    )
    argument_output_path = args.argument_output.resolve()
    argument_cases_path = args.argument_cases.resolve()
    argument_result = analyze_argument_errors(
        eval_root=args.eval_root.resolve(),
        summary_path=argument_output_path,
        cases_path=argument_cases_path,
        base_run=args.base_run,
        sft_run=args.sft_run,
        splits=tuple(args.splits),
    )
    wrong_value_output_path = args.wrong_value_output.resolve()
    wrong_value_spotcheck_path = args.wrong_value_spotcheck.resolve()
    wrong_value_result = analyze_wrong_values(
        argument_cases_path=argument_cases_path,
        canonical_root=args.canonical_root.resolve(),
        summary_path=wrong_value_output_path,
        spotcheck_path=wrong_value_spotcheck_path,
        annotations_path=args.wrong_value_annotations.resolve(),
        splits=tuple(args.splits),
        spotcheck_per_split=args.spotcheck_per_split,
        seed=args.spotcheck_seed,
    )
    print("Day 3 error analysis Phases 1-4 complete")
    for split, summary in result["splits"].items():
        transitions = summary["counts"]["transitions"]
        print(
            f"{split}: samples={summary['counts']['samples']}, "
            f"fixed={transitions['fixed_by_sft']}, "
            f"regressed={transitions['regressed_by_sft']}, "
            f"still_wrong={transitions['still_wrong']}"
        )
    print(f"transition summary: {output_path}")
    for split, summary in error_result["splits"].items():
        print(
            f"{split} SFT failures: samples={summary['failure_samples']}, "
            f"error_occurrences={summary['error_occurrences']}"
        )
    print(f"error summary: {error_output_path}")
    for split, summary in transition_error_result["splits"].items():
        repaired = summary["repaired_base_errors"]
        regressed = summary["regression_sft_errors"]
        print(
            f"{split} transition errors: "
            f"repaired={repaired['transition_samples']}, "
            f"regressed={regressed['transition_samples']}"
        )
    print(f"transition error summary: {transition_error_output_path}")
    for split, summary in argument_result["splits"].items():
        print(
            f"{split} argument mismatches: "
            f"samples={summary['argument_mismatch_samples']}, "
            f"clean={summary['clean_argument_failures']}, "
            f"mixed={summary['mixed_argument_failures']}"
        )
    print(f"argument summary: {argument_output_path}")
    print(f"argument cases: {argument_cases_path}")
    spotcheck = wrong_value_result["spotcheck"]
    print(
        "wrong-value spot check: "
        f"samples={spotcheck['total_samples']}, "
        f"reviewed={spotcheck['reviewed_samples']}, "
        f"pending={spotcheck['pending_samples']}"
    )
    print(f"wrong-value summary: {wrong_value_output_path}")
    print(f"wrong-value spot check cases: {wrong_value_spotcheck_path}")


if __name__ == "__main__":
    main()

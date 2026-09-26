"""Run selected BFCL v4 single-turn categories through LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.eval.bfcl_common import (
    BFCLAdapterError,
    DEFAULT_ADAPTER_PATH,
    DEFAULT_CATEGORIES,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_DIR,
    dataset_path,
    load_jsonl,
    parse_categories,
    raw_prediction_path,
    require_unique_ids,
    resolve_bfcl_data_dir,
)
from src.eval.inference import InferenceConfig, InferenceRunner


@dataclass(frozen=True)
class BFCLInferenceRequest:
    entry_id: str
    category: str
    messages: list[dict[str, str]]
    system: str | None
    tools: str


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BFCLAdapterError(f"{field} must be a string")
    return value


def bfcl_to_request(entry: dict[str, Any], category: str) -> BFCLInferenceRequest:
    """Map one official BFCL v4 entry to LLaMA-Factory's chat contract."""
    entry_id = _require_string(entry.get("id"), "id")
    question = entry.get("question")
    if not isinstance(question, list) or len(question) != 1:
        raise BFCLAdapterError(
            f"{entry_id}.question must contain exactly one turn for this adapter"
        )
    turn = question[0]
    if not isinstance(turn, list) or not turn:
        raise BFCLAdapterError(f"{entry_id}.question[0] must be a non-empty array")

    system: str | None = None
    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(turn):
        if not isinstance(raw_message, dict):
            raise BFCLAdapterError(f"{entry_id}.question[0][{index}] must be an object")
        role = _require_string(raw_message.get("role"), f"{entry_id}.role")
        content = _require_string(raw_message.get("content"), f"{entry_id}.content")
        if role == "system":
            if index != 0 or system is not None:
                raise BFCLAdapterError(
                    f"{entry_id}: only one leading system message is supported"
                )
            system = content
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
        elif role == "tool":
            messages.append({"role": "observation", "content": content})
        else:
            raise BFCLAdapterError(f"{entry_id}: unsupported message role {role!r}")
    if not messages or messages[-1]["role"] != "user":
        raise BFCLAdapterError(f"{entry_id}: turn must end with a user message")

    functions = entry.get("function")
    if not isinstance(functions, list) or not functions:
        raise BFCLAdapterError(f"{entry_id}.function must be a non-empty array")
    wrapped_tools: list[dict[str, Any]] = []
    for index, function in enumerate(functions):
        if not isinstance(function, dict):
            raise BFCLAdapterError(f"{entry_id}.function[{index}] must be an object")
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise BFCLAdapterError(
                f"{entry_id}.function[{index}].name must be a non-empty string"
            )
        wrapped_tools.append({"type": "function", "function": function})

    return BFCLInferenceRequest(
        entry_id=entry_id,
        category=category,
        messages=messages,
        system=system,
        tools=json.dumps(wrapped_tools, ensure_ascii=False, separators=(",", ":")),
    )


class BFCLInferenceRunner:
    """BFCL-specific record adapter around one reusable model instance."""

    def __init__(self, inference_runner: InferenceRunner) -> None:
        self.inference_runner = inference_runner

    def predict(self, entry: dict[str, Any], category: str) -> dict[str, Any]:
        request = bfcl_to_request(entry, category)
        started = perf_counter()
        response = self.inference_runner.generate(
            request.messages,
            system=request.system,
            tools=request.tools,
            sample_id=request.entry_id,
        )
        latency = perf_counter() - started
        return {
            "id": request.entry_id,
            "test_category": category,
            "raw_prediction": response.response_text,
            "generation": {
                "prompt_tokens": response.prompt_length,
                "response_tokens": response.response_length,
                "finish_reason": response.finish_reason,
                "latency_seconds": latency,
            },
        }


def _load_completed_ids(path: Path, category: str) -> set[str]:
    if not path.exists():
        return set()
    entries = load_jsonl(path)
    ids = require_unique_ids(entries, source=path)
    for index, entry in enumerate(entries, start=1):
        if entry.get("test_category") != category:
            raise BFCLAdapterError(
                f"{path}:{index}: test_category must be {category!r}"
            )
        if not isinstance(entry.get("raw_prediction"), str):
            raise BFCLAdapterError(f"{path}:{index}: raw_prediction must be a string")
    return set(ids)


def run_category(
    runner: BFCLInferenceRunner,
    *,
    category: str,
    data_dir: Path,
    raw_dir: Path,
    max_samples: int | None = None,
    overwrite: bool = False,
    progress_every: int = 10,
) -> tuple[int, int]:
    """Generate one category with durable per-record resume semantics."""
    if max_samples is not None and max_samples <= 0:
        raise BFCLAdapterError("max_samples must be positive")
    if progress_every < 0:
        raise BFCLAdapterError("progress_every must be non-negative")

    source_path = dataset_path(data_dir, category)
    if not source_path.is_file():
        raise FileNotFoundError(f"BFCL dataset not found: {source_path}")
    entries = load_jsonl(source_path)
    source_ids = require_unique_ids(entries, source=source_path)
    output_path = raw_prediction_path(raw_dir, category)
    completed_ids = set() if overwrite else _load_completed_ids(output_path, category)
    unknown_ids = completed_ids - set(source_ids)
    if unknown_ids:
        preview = ", ".join(repr(item) for item in sorted(unknown_ids)[:5])
        raise BFCLAdapterError(
            f"{output_path}: prediction ids absent from BFCL data: {preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    append = not overwrite and output_path.exists()
    needs_separator = (
        append
        and output_path.stat().st_size > 0
        and not output_path.read_bytes().endswith(b"\n")
    )
    mode = "a" if append else "w"
    generated = 0
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        if needs_separator:
            output.write("\n")
        for entry in entries:
            entry_id = entry["id"]
            if entry_id in completed_ids:
                continue
            if max_samples is not None and generated >= max_samples:
                break
            prediction = runner.predict(entry, category)
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            generated += 1
            completed = len(completed_ids) + generated
            if progress_every and (generated == 1 or generated % progress_every == 0):
                print(
                    f"progress: {category} {completed}/{len(entries)} "
                    f"(generated_this_run: {generated})",
                    flush=True,
                )
    completed = len(completed_ids) + generated
    if progress_every and generated and generated % progress_every != 0:
        print(
            f"progress: {category} {completed}/{len(entries)} "
            f"(generated_this_run: {generated})",
            flush=True,
        )
    return generated, completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw Qwen3 predictions for the selected BFCL v4 categories."
    )
    parser.add_argument(
        "--bfcl-data-dir",
        type=Path,
        default=None,
        help="Directory containing BFCL_v4_*.json package data.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated concrete BFCL categories.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--model-name-or-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Merged/local base model path or Hugging Face id.",
    )
    parser.add_argument(
        "--adapter-name-or-path",
        default=str(DEFAULT_ADAPTER_PATH),
        help="LoRA adapter path; pass an empty string to disable it.",
    )
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--infer-backend", default="huggingface")
    parser.add_argument("--quantization-bit", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=None,
        help="For smoke tests only; default runs every entry.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.categories)
    data_dir = resolve_bfcl_data_dir(args.bfcl_data_dir)
    adapter = args.adapter_name_or_path or None
    config = InferenceConfig(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=adapter,
        template=args.template,
        infer_backend=args.infer_backend,
        quantization_bit=args.quantization_bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    runner = BFCLInferenceRunner(InferenceRunner(config, verbose=args.verbose))

    total_generated = 0
    for category in categories:
        generated, completed = run_category(
            runner,
            category=category,
            data_dir=data_dir,
            raw_dir=args.raw_dir.resolve(),
            max_samples=args.max_samples_per_category,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
        total_generated += generated
        print(f"category_complete: {category} {completed}", flush=True)

    print("BFCL raw inference complete")
    print(f"generated_this_run: {total_generated}")
    print(f"raw_predictions: {args.raw_dir.resolve()}")


if __name__ == "__main__":
    main()

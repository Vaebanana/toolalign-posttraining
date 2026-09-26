"""Run canonical tool-use samples through LLaMA-Factory's ChatModel.

This module is intentionally a thin inference adapter.  LLaMA-Factory owns
model loading, quantization, prompt templating, generation and decoding.  This
module only maps the project's canonical records to ``ChatModel.chat`` calls
and persists raw predictions for the downstream parser/evaluator.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dev_seen.jsonl"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "eval" / "base" / "dev_seen" / "predictions.jsonl"
)
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "tool": "observation",
}


class InferenceError(ValueError):
    """A sample cannot be inferred without changing its meaning."""


class ChatResponse(Protocol):
    """The part of LLaMA-Factory's Response contract used by this project."""

    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: str


class ChatBackend(Protocol):
    """Structural type for LLaMA-Factory ChatModel and test doubles."""

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **input_kwargs: Any,
    ) -> list[ChatResponse]: ...


@dataclass(frozen=True)
class InferenceConfig:
    """Model and generation settings shared by Base and SFT evaluation."""

    model_name_or_path: str = DEFAULT_MODEL
    adapter_name_or_path: str | None = None
    template: str = "qwen3_nothink"
    infer_backend: str = "huggingface"
    quantization_bit: int | None = 4
    trust_remote_code: bool = True
    max_new_tokens: int = 512
    do_sample: bool = False

    def to_chat_model_args(self) -> dict[str, Any]:
        """Return arguments accepted by LLaMA-Factory ``ChatModel``."""
        if not self.model_name_or_path:
            raise InferenceError("model_name_or_path must be non-empty")
        if self.quantization_bit not in {None, 4, 8}:
            raise InferenceError("quantization_bit must be 4, 8 or None")
        if self.max_new_tokens <= 0:
            raise InferenceError("max_new_tokens must be positive")

        args: dict[str, Any] = {
            "model_name_or_path": self.model_name_or_path,
            "template": self.template,
            "infer_backend": self.infer_backend,
            "trust_remote_code": self.trust_remote_code,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.adapter_name_or_path is not None:
            args["adapter_name_or_path"] = self.adapter_name_or_path
        if self.quantization_bit is not None:
            args["quantization_bit"] = self.quantization_bit
        return args


@dataclass(frozen=True)
class InferenceRequest:
    """One canonical sample mapped to the ChatModel input contract."""

    sample_id: str
    messages: list[dict[str, str]]
    system: str
    tools: str


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceError(f"{field} must be an object")
    return value


def _require_array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise InferenceError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InferenceError(f"{field} must be a string")
    return value


def canonical_to_request(sample: dict[str, Any]) -> InferenceRequest:
    """Map one canonical record to LLaMA-Factory's chat input.

    This function validates and maps fields only.  It deliberately does not
    repair, filter or reinterpret malformed samples.
    """
    sample_id = _require_string(sample.get("sample_id"), "sample_id")
    raw_messages = _require_array(sample.get("messages"), f"{sample_id}.messages")
    tools = _require_array(sample.get("tools"), f"{sample_id}.tools")
    _require_object(sample.get("assistant"), f"{sample_id}.assistant")

    system_messages: list[str] = []
    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(raw_messages):
        message = _require_object(
            raw_message,
            f"{sample_id}.messages[{index}]",
        )
        role = _require_string(
            message.get("role"),
            f"{sample_id}.messages[{index}].role",
        )
        content = _require_string(
            message.get("content"),
            f"{sample_id}.messages[{index}].content",
        )
        if role == "system":
            system_messages.append(content)
            continue
        try:
            mapped_role = ROLE_MAP[role]
        except KeyError as error:
            raise InferenceError(
                f"{sample_id}.messages[{index}].role is unsupported: {role!r}"
            ) from error
        messages.append({"role": mapped_role, "content": content})

    if len(system_messages) != 1:
        raise InferenceError(
            f"{sample_id} must contain exactly one system message; "
            f"got {len(system_messages)}"
        )
    if not messages:
        raise InferenceError(f"{sample_id} has no non-system messages")

    return InferenceRequest(
        sample_id=sample_id,
        messages=messages,
        system=system_messages[0],
        tools=json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
    )


class InferenceRunner:
    """Thin, reusable wrapper around LLaMA-Factory ``ChatModel``."""

    def __init__(
        self,
        config: InferenceConfig,
        chat_model: ChatBackend | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self._framework_diagnostics = io.StringIO()
        model_args = config.to_chat_model_args()
        if chat_model is None:
            def load_chat_model() -> ChatBackend:
                try:
                    from llamafactory.chat import ChatModel
                except ImportError as error:
                    raise RuntimeError(
                        "LLaMA-Factory is not importable. Run this command in the "
                        "environment where llamafactory-cli is installed."
                    ) from error
                return ChatModel(model_args)

            chat_model = self._run_framework_call(load_chat_model)
        self.chat_model = chat_model

    def _run_framework_call(self, function: Any) -> Any:
        """Hide framework chatter on success and replay it on failure."""
        if self.verbose:
            return function()
        try:
            with (
                redirect_stdout(self._framework_diagnostics),
                redirect_stderr(self._framework_diagnostics),
            ):
                return function()
        except BaseException:
            diagnostics = self._framework_diagnostics.getvalue().strip()
            if diagnostics:
                print("LLaMA-Factory diagnostic output:", file=sys.stderr)
                print(diagnostics, file=sys.stderr)
            raise

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Generate and return one raw prediction record."""
        request = canonical_to_request(sample)
        response = self.generate(
            request.messages,
            system=request.system,
            tools=request.tools,
            sample_id=request.sample_id,
        )

        record: dict[str, Any] = {
            "sample_id": request.sample_id,
            "gold": sample["assistant"],
            "raw_prediction": response.response_text,
            "generation": {
                "prompt_tokens": response.prompt_length,
                "response_tokens": response.response_length,
                "finish_reason": response.finish_reason,
            },
        }
        # Preserve analysis labels without coupling inference to their shape.
        for field in ("source", "metadata"):
            if field in sample:
                record[field] = sample[field]
        return record

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None,
        tools: str,
        sample_id: str = "request",
    ) -> ChatResponse:
        """Generate one response from an already-adapted chat request.

        Keeping this small public boundary lets benchmark adapters reuse the
        same loaded LLaMA-Factory model without pretending their records use
        the project's canonical training schema.
        """
        responses = self._run_framework_call(
            lambda: self.chat_model.chat(
                messages,
                system=system,
                tools=tools,
            )
        )
        if len(responses) != 1:
            raise RuntimeError(
                f"{sample_id}: expected one response, got {len(responses)}"
            )
        return responses[0]


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield validated JSON objects with their source line numbers."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise InferenceError(f"{path}:{line_number}: invalid JSON") from error
            yield line_number, _require_object(value, f"{path}:{line_number}")


def load_sample_ids(path: Path, *, label: str) -> set[str]:
    """Load unique sample IDs from a JSONL file."""
    sample_ids: set[str] = set()
    for line_number, record in read_jsonl(path):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise InferenceError(
                f"{path}:{line_number}: {label} sample_id must be a non-empty string"
            )
        if sample_id in sample_ids:
            raise InferenceError(
                f"{path}:{line_number}: duplicate {label} sample_id {sample_id!r}"
            )
        sample_ids.add(sample_id)
    return sample_ids


def load_completed_prediction_ids(output_path: Path) -> set[str]:
    """Validate an existing prediction file and return completed sample IDs."""
    if not output_path.exists():
        return set()
    if not output_path.is_file():
        raise InferenceError(f"prediction output is not a file: {output_path}")
    return load_sample_ids(output_path, label="prediction")


def run_jsonl(
    runner: InferenceRunner,
    input_path: Path,
    output_path: Path,
    *,
    max_samples: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    progress_every: int = 0,
) -> int:
    """Generate pending records, resuming by sample ID unless overwritten."""
    if max_samples is not None and max_samples <= 0:
        raise InferenceError("max_samples must be positive")
    if start_index < 0:
        raise InferenceError("start_index must be non-negative")
    if progress_every < 0:
        raise InferenceError("progress_every must be non-negative")
    if input_path.resolve() == output_path.resolve():
        raise InferenceError("input_path and output_path must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset not found: {input_path}")

    input_ids = load_sample_ids(input_path, label="input")
    completed_ids = set() if overwrite else load_completed_prediction_ids(output_path)
    unknown_ids = completed_ids - input_ids
    if unknown_ids:
        preview = ", ".join(repr(value) for value in sorted(unknown_ids)[:5])
        raise InferenceError(
            f"existing predictions contain sample_id values absent from input: {preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    append = not overwrite and output_path.exists()
    needs_separator = (
        append
        and output_path.stat().st_size > 0
        and not output_path.read_bytes().endswith(b"\n")
    )
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        if needs_separator:
            output.write("\n")
        for row_index, (line_number, sample) in enumerate(read_jsonl(input_path)):
            if row_index < start_index:
                continue
            sample_id = sample["sample_id"]
            if sample_id in completed_ids:
                continue
            if max_samples is not None and count >= max_samples:
                break
            try:
                prediction = runner.predict(sample)
            except Exception as error:
                raise RuntimeError(
                    f"inference failed at {input_path}:{line_number}"
                ) from error
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            if progress_every and (count == 1 or count % progress_every == 0):
                completed = len(completed_ids) + count
                print(
                    f"progress: {completed}/{len(input_ids)} "
                    f"(generated_this_run: {count})",
                    flush=True,
                )
    if progress_every and count and count % progress_every != 0:
        completed = len(completed_ids) + count
        print(
            f"progress: {completed}/{len(input_ids)} "
            f"(generated_this_run: {count})",
            flush=True,
        )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical tool-use samples with LLaMA-Factory ChatModel."
        )
    )
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL,
        help=f"Base model path or Hub id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--adapter-name-or-path",
        default=None,
        help="Optional LoRA adapter path; omit for Base evaluation.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Canonical JSONL input (default: data/processed/dev_seen.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Raw prediction JSONL output.",
    )
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--infer-backend", default="huggingface")
    parser.add_argument(
        "--quantization-bit",
        type=int,
        choices=(4, 8),
        default=4,
        help="bitsandbytes inference precision (default: 4).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of records to run after --start-index.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based input row at which to start (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Discard an existing output and regenerate it. By default, existing "
            "sample_id values are validated and skipped."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one concise progress line every N new samples; 0 disables it.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code when loading the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show LLaMA-Factory logs and per-10-sample progress.",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        template=args.template,
        infer_backend=args.infer_backend,
        quantization_bit=args.quantization_bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    runner = InferenceRunner(config, verbose=args.verbose)
    count = run_jsonl(
        runner,
        args.input.resolve(),
        args.output.resolve(),
        max_samples=args.max_samples,
        start_index=args.start_index,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    total = len(load_completed_prediction_ids(args.output.resolve()))
    print("LLaMA-Factory inference complete")
    print(f"generated_this_run: {count}")
    print(f"total_predictions: {total}")
    print(f"predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()

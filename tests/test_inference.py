"""Contract tests for the thin LLaMA-Factory inference wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from src.eval.inference import (
    InferenceConfig,
    InferenceError,
    InferenceRunner,
    canonical_to_request,
    load_completed_prediction_ids,
    run_jsonl,
)


def canonical_sample(sample_id: str = "sample_1") -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source": "test",
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "What is the weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "assistant": {
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                }
            ],
        },
        "metadata": {"kind": "single"},
    }


@dataclass
class FakeResponse:
    response_text: str = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
    response_length: int = 12
    prompt_length: int = 42
    finish_reason: str = "stop"


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **input_kwargs: Any,
    ) -> list[FakeResponse]:
        self.calls.append(
            {"messages": messages, "system": system, "tools": tools}
        )
        return [FakeResponse()]


class NoisyFakeChatModel(FakeChatModel):
    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **input_kwargs: Any,
    ) -> list[FakeResponse]:
        print("framework stdout")
        print("framework stderr", file=__import__("sys").stderr)
        return super().chat(messages, system, tools, **input_kwargs)


class InferenceTests(unittest.TestCase):
    def test_config_matches_llamafactory_contract(self) -> None:
        args = InferenceConfig(adapter_name_or_path="adapter").to_chat_model_args()
        self.assertEqual(args["template"], "qwen3_nothink")
        self.assertEqual(args["infer_backend"], "huggingface")
        self.assertEqual(args["quantization_bit"], 4)
        self.assertEqual(args["adapter_name_or_path"], "adapter")
        self.assertFalse(args["do_sample"])

    def test_canonical_mapping_keeps_system_messages_and_tools_separate(self) -> None:
        request = canonical_to_request(canonical_sample())
        self.assertEqual(
            request.messages,
            [{"role": "user", "content": "What is the weather?"}],
        )
        self.assertEqual(request.system, "Use the tools.")
        self.assertEqual(json.loads(request.tools)[0]["function"]["name"], "get_weather")

    def test_runner_calls_chatmodel_and_records_raw_response(self) -> None:
        backend = FakeChatModel()
        runner = InferenceRunner(InferenceConfig(), chat_model=backend)
        result = runner.predict(canonical_sample())

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["system"], "Use the tools.")
        self.assertEqual(result["sample_id"], "sample_1")
        self.assertEqual(result["gold"], canonical_sample()["assistant"])
        self.assertEqual(result["raw_prediction"], FakeResponse.response_text)
        self.assertEqual(result["generation"]["prompt_tokens"], 42)

    def test_runner_is_quiet_by_default(self) -> None:
        runner = InferenceRunner(InferenceConfig(), chat_model=NoisyFakeChatModel())
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            runner.predict(canonical_sample())

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_verbose_runner_preserves_framework_output(self) -> None:
        runner = InferenceRunner(
            InferenceConfig(),
            chat_model=NoisyFakeChatModel(),
            verbose=True,
        )
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            runner.predict(canonical_sample())

        self.assertIn("framework stdout", stdout.getvalue())
        self.assertIn("framework stderr", stderr.getvalue())

    def test_invalid_sample_is_rejected_not_repaired(self) -> None:
        sample = canonical_sample()
        sample["messages"] = [{"role": "user", "content": "hello"}]
        with self.assertRaisesRegex(InferenceError, "exactly one system"):
            canonical_to_request(sample)

    def test_jsonl_slice_is_stable_and_persisted(self) -> None:
        backend = FakeChatModel()
        runner = InferenceRunner(InferenceConfig(), chat_model=backend)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            with input_path.open("w", encoding="utf-8") as file:
                for index in range(3):
                    file.write(json.dumps(canonical_sample(f"sample_{index}")) + "\n")

            count = run_jsonl(
                runner,
                input_path,
                output_path,
                start_index=1,
                max_samples=1,
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 1)
        self.assertEqual([row["sample_id"] for row in rows], ["sample_1"])

    def test_jsonl_resume_skips_existing_sample_ids(self) -> None:
        backend = FakeChatModel()
        runner = InferenceRunner(InferenceConfig(), chat_model=backend)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(canonical_sample(f"sample_{index}")) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            first = run_jsonl(runner, input_path, output_path, max_samples=1)
            resumed = run_jsonl(runner, input_path, output_path)
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first, 1)
        self.assertEqual(resumed, 2)
        self.assertEqual(
            [row["sample_id"] for row in rows],
            ["sample_0", "sample_1", "sample_2"],
        )

    def test_jsonl_progress_is_concise_and_resume_aware(self) -> None:
        runner = InferenceRunner(InferenceConfig(), chat_model=FakeChatModel())
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(canonical_sample(f"sample_{index}")) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            run_jsonl(runner, input_path, output_path, max_samples=1)
            stdout = StringIO()
            with redirect_stdout(stdout):
                run_jsonl(
                    runner,
                    input_path,
                    output_path,
                    progress_every=1,
                )

        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                "progress: 2/3 (generated_this_run: 1)",
                "progress: 3/3 (generated_this_run: 2)",
            ],
        )

    def test_jsonl_resume_rejects_unknown_existing_sample_id(self) -> None:
        runner = InferenceRunner(InferenceConfig(), chat_model=FakeChatModel())
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                json.dumps(canonical_sample("known")) + "\n",
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps({"sample_id": "unknown"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InferenceError, "absent from input"):
                run_jsonl(runner, input_path, output_path)

    def test_jsonl_overwrite_replaces_existing_predictions(self) -> None:
        runner = InferenceRunner(InferenceConfig(), chat_model=FakeChatModel())
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                json.dumps(canonical_sample("sample_0")) + "\n",
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps({"sample_id": "old"}) + "\n",
                encoding="utf-8",
            )
            count = run_jsonl(
                runner,
                input_path,
                output_path,
                overwrite=True,
            )
            completed = load_completed_prediction_ids(output_path)

        self.assertEqual(count, 1)
        self.assertEqual(completed, {"sample_0"})


if __name__ == "__main__":
    unittest.main()

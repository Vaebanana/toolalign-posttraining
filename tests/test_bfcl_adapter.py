"""Contract tests for the BFCL v4 inference and result adapters."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.bfcl_common import BFCLAdapterError, raw_prediction_path
from src.eval.bfcl_convert import convert_category, to_official_record
from src.eval.bfcl_inference import (
    BFCLInferenceRunner,
    bfcl_to_request,
    run_category,
)
from src.eval.inference import InferenceConfig, InferenceRunner


def bfcl_entry(entry_id: str = "parallel_0", *, system: bool = False) -> dict[str, Any]:
    messages = [{"role": "user", "content": "Get weather for Paris."}]
    if system:
        messages.insert(0, {"role": "system", "content": "Use tools precisely."})
    return {
        "id": entry_id,
        "question": [messages],
        "function": [
            {
                "name": "weather.get",
                "description": "Get weather.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }


@dataclass
class FakeResponse:
    response_text: str = (
        '<tool_call>\n{"name":"weather.get","arguments":{"city":"Paris"}}\n'
        "</tool_call>"
    )
    response_length: int = 20
    prompt_length: int = 100
    finish_reason: str = "stop"


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **kwargs: Any,
    ) -> list[FakeResponse]:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        return [FakeResponse()]


class BFCLAdapterTests(unittest.TestCase):
    def test_bfcl_request_wraps_functions_and_preserves_schema(self) -> None:
        request = bfcl_to_request(bfcl_entry(), "parallel")

        self.assertIsNone(request.system)
        self.assertEqual(request.messages, [{"role": "user", "content": "Get weather for Paris."}])
        tools = json.loads(request.tools)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "weather.get")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "dict")

    def test_bfcl_request_separates_leading_system_message(self) -> None:
        request = bfcl_to_request(bfcl_entry(system=True), "live_simple")
        self.assertEqual(request.system, "Use tools precisely.")
        self.assertEqual(len(request.messages), 1)

    def test_multiturn_entry_is_rejected_by_scoped_adapter(self) -> None:
        entry = bfcl_entry()
        entry["question"].append([{"role": "user", "content": "Next turn"}])
        with self.assertRaisesRegex(BFCLAdapterError, "exactly one turn"):
            bfcl_to_request(entry, "parallel")

    def test_inference_is_resumable_and_records_raw_output(self) -> None:
        backend = FakeChatModel()
        runner = BFCLInferenceRunner(
            InferenceRunner(InferenceConfig(), chat_model=backend)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            raw_dir = root / "raw"
            data_dir.mkdir()
            source = data_dir / "BFCL_v4_parallel.json"
            source.write_text(
                "".join(
                    json.dumps(bfcl_entry(f"parallel_{index}")) + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            first = run_category(
                runner,
                category="parallel",
                data_dir=data_dir,
                raw_dir=raw_dir,
                max_samples=1,
                progress_every=0,
            )
            resumed = run_category(
                runner,
                category="parallel",
                data_dir=data_dir,
                raw_dir=raw_dir,
                progress_every=0,
            )
            rows = [
                json.loads(line)
                for line in raw_prediction_path(raw_dir, "parallel")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(first, (1, 1))
        self.assertEqual(resumed, (1, 2))
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(rows[0]["raw_prediction"], FakeResponse.response_text)
        self.assertGreaterEqual(rows[0]["generation"]["latency_seconds"], 0)

    def test_official_conversion_preserves_raw_prediction(self) -> None:
        prediction = {
            "id": "parallel_0",
            "test_category": "parallel",
            "raw_prediction": FakeResponse.response_text,
            "generation": {
                "prompt_tokens": 100,
                "response_tokens": 20,
                "latency_seconds": 0.25,
            },
        }
        record = to_official_record(
            prediction,
            category="parallel",
            source=Path("raw.jsonl"),
            line=1,
        )
        self.assertEqual(record["result"], FakeResponse.response_text)
        self.assertEqual(record["input_token_count"], 100)
        self.assertEqual(record["latency"], 0.25)

    def test_conversion_requires_complete_ids_unless_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            raw_dir = root / "raw"
            result_dir = root / "result"
            data_dir.mkdir()
            raw_dir.mkdir()
            (data_dir / "BFCL_v4_parallel.json").write_text(
                json.dumps(bfcl_entry("parallel_0"))
                + "\n"
                + json.dumps(bfcl_entry("parallel_1"))
                + "\n",
                encoding="utf-8",
            )
            raw_prediction_path(raw_dir, "parallel").write_text(
                json.dumps(
                    {
                        "id": "parallel_0",
                        "test_category": "parallel",
                        "raw_prediction": FakeResponse.response_text,
                        "generation": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BFCLAdapterError, "incomplete category"):
                convert_category(
                    category="parallel",
                    data_dir=data_dir,
                    raw_dir=raw_dir,
                    result_dir=result_dir,
                    registry_name="local-FC",
                )
            output = convert_category(
                category="parallel",
                data_dir=data_dir,
                raw_dir=raw_dir,
                result_dir=result_dir,
                registry_name="local-FC",
                allow_partial=True,
            )

        self.assertEqual(
            output.parts[-3:],
            ("local-FC", "non_live", "BFCL_v4_parallel_result.json"),
        )


if __name__ == "__main__":
    unittest.main()

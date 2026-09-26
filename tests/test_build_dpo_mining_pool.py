"""Tests for deterministic train-side DPO mining-pool construction."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_dpo_mining_pool import build_mining_pool
from src.data.export_llamafactory import export_sample


def canonical(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "source": "test",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"query {sample_id}"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "A",
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "assistant": {
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "A", "arguments": {}},
                }
            ],
        },
        "metadata": {},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class DpoMiningPoolTests(unittest.TestCase):
    def test_excludes_sft_ids_and_is_deterministic(self) -> None:
        records = [canonical(f"s{index}") for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            llama_path = root / "llama.jsonl"
            sft_manifest_path = root / "sft.manifest.json"
            write_jsonl(canonical_path, records)
            write_jsonl(llama_path, [export_sample(record) for record in records])
            llama_hash = hashlib.sha256(llama_path.read_bytes()).hexdigest()
            selected_lines = [2, 5]
            selected_hash = hashlib.sha256(b"2\n5").hexdigest()
            sft_manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "toolalign-sft-subset-v1",
                        "source_sample_count": 6,
                        "source_sha256": llama_hash,
                        "sample_count": 2,
                        "selected_line_numbers_sha256": selected_hash,
                        "selected_line_numbers": selected_lines,
                    }
                ),
                encoding="utf-8",
            )
            output_a = root / "pool_a.jsonl"
            output_b = root / "pool_b.jsonl"
            manifest_a = build_mining_pool(
                canonical_path,
                llama_path,
                sft_manifest_path,
                output_a,
                root / "pool_a.manifest.json",
                sample_count=3,
                seed=11,
            )
            manifest_b = build_mining_pool(
                canonical_path,
                llama_path,
                sft_manifest_path,
                output_b,
                root / "pool_b.manifest.json",
                sample_count=3,
                seed=11,
            )

        self.assertEqual(manifest_a["exclusion"]["sample_ids"], ["s1", "s4"])
        self.assertEqual(manifest_a["sample_ids"], manifest_b["sample_ids"])
        self.assertFalse(set(manifest_a["sample_ids"]) & {"s1", "s4"})
        self.assertEqual(manifest_a["eligible_sample_count"], 4)
        self.assertTrue(manifest_a["canonical_llamafactory_full_alignment_verified"])
        self.assertFalse(manifest_a["preference_pairs_generated"])

    def test_rejects_misaligned_llamafactory_source(self) -> None:
        records = [canonical("s0"), canonical("s1")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            llama_path = root / "llama.jsonl"
            sft_manifest_path = root / "sft.manifest.json"
            write_jsonl(canonical_path, records)
            exported = [export_sample(record) for record in reversed(records)]
            write_jsonl(llama_path, exported)
            llama_hash = hashlib.sha256(llama_path.read_bytes()).hexdigest()
            sft_manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "toolalign-sft-subset-v1",
                        "source_sample_count": 2,
                        "source_sha256": llama_hash,
                        "sample_count": 1,
                        "selected_line_numbers_sha256": hashlib.sha256(b"1").hexdigest(),
                        "selected_line_numbers": [1],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "misaligned"):
                build_mining_pool(
                    canonical_path,
                    llama_path,
                    sft_manifest_path,
                    root / "pool.jsonl",
                    root / "pool.manifest.json",
                    sample_count=1,
                )

    def test_excludes_a_previous_mining_pool(self) -> None:
        records = [canonical(f"s{index}") for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            llama_path = root / "llama.jsonl"
            sft_manifest_path = root / "sft.manifest.json"
            write_jsonl(canonical_path, records)
            write_jsonl(llama_path, [export_sample(record) for record in records])
            llama_hash = hashlib.sha256(llama_path.read_bytes()).hexdigest()
            sft_manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "toolalign-sft-subset-v1",
                        "source_sample_count": 10,
                        "source_sha256": llama_hash,
                        "sample_count": 1,
                        "selected_line_numbers_sha256": hashlib.sha256(
                            b"1"
                        ).hexdigest(),
                        "selected_line_numbers": [1],
                    }
                ),
                encoding="utf-8",
            )
            first_manifest_path = root / "pool_a.manifest.json"
            first = build_mining_pool(
                canonical_path,
                llama_path,
                sft_manifest_path,
                root / "pool_a.jsonl",
                first_manifest_path,
                sample_count=3,
                seed=21,
            )
            second = build_mining_pool(
                canonical_path,
                llama_path,
                sft_manifest_path,
                root / "pool_b.jsonl",
                root / "pool_b.manifest.json",
                sample_count=3,
                seed=22,
                previous_mining_manifest_paths=[first_manifest_path],
            )

        self.assertFalse(set(first["sample_ids"]) & set(second["sample_ids"]))
        self.assertFalse(set(second["sample_ids"]) & {"s0"})
        self.assertEqual(second["eligible_sample_count"], 6)
        self.assertEqual(second["excluded_sample_count_total"], 4)
        self.assertEqual(
            second["previous_mining_exclusions"][0]["sample_ids"],
            first["sample_ids"],
        )

    def test_restricts_sampling_to_rule_triage_auto_accept(self) -> None:
        records = [canonical(f"s{index}") for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.jsonl"
            llama_path = root / "llama.jsonl"
            sft_manifest_path = root / "sft.manifest.json"
            prescreen_path = root / "prescreen.manifest.json"
            write_jsonl(canonical_path, records)
            write_jsonl(llama_path, [export_sample(record) for record in records])
            canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
            llama_hash = hashlib.sha256(llama_path.read_bytes()).hexdigest()
            sft_manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "toolalign-sft-subset-v1",
                        "source_sample_count": 8,
                        "source_sha256": llama_hash,
                        "sample_count": 1,
                        "selected_line_numbers_sha256": hashlib.sha256(b"1").hexdigest(),
                        "selected_line_numbers": [1],
                    }
                ),
                encoding="utf-8",
            )
            candidate_lines = [2, 4, 6, 8]
            candidate_ids = ["s1", "s3", "s5", "s7"]
            prescreen_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "toolalign-dpo-groundability-rule-triage-v1",
                        "status": "rule_triage_complete_no_rollout",
                        "source": {
                            "canonical_sha256": canonical_hash,
                            "llamafactory_sha256": llama_hash,
                        },
                        "rollout_priority_selection": {
                            "selection_label": "AUTO_ACCEPT",
                            "semantically_grounded": False,
                            "source_line_numbers_sha256": hashlib.sha256(
                                b"2\n4\n6\n8"
                            ).hexdigest(),
                            "sample_ids_sha256": hashlib.sha256(
                                b"s1\ns3\ns5\ns7"
                            ).hexdigest(),
                            "source_line_numbers": candidate_lines,
                            "sample_ids": candidate_ids,
                        },
                        "operations": {
                            "rollout_started": False,
                            "predictions_generated": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_mining_pool(
                canonical_path,
                llama_path,
                sft_manifest_path,
                root / "pool.jsonl",
                root / "pool.manifest.json",
                sample_count=3,
                seed=45,
                candidate_prescreen_manifest_path=prescreen_path,
            )

        self.assertTrue(set(manifest["sample_ids"]).issubset(candidate_ids))
        self.assertEqual(manifest["eligible_sample_count"], 4)
        self.assertEqual(manifest["eligible_before_candidate_filter_count"], 7)
        self.assertEqual(
            manifest["candidate_prescreen"]["selection_label"], "AUTO_ACCEPT"
        )
        self.assertFalse(manifest["candidate_prescreen"]["semantically_grounded"])


if __name__ == "__main__":
    unittest.main()

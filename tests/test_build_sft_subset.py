"""Tests for deterministic SFT subset construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_sft_subset import SubsetError, build_subset


class BuildSftSubsetTests(unittest.TestCase):
    def _source(self, directory: Path, count: int = 20) -> Path:
        path = directory / "train.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for index in range(count):
                file.write(json.dumps({"row": index}) + "\n")
        return path

    def test_same_seed_produces_identical_ordered_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source = self._source(directory)
            first = directory / "first.jsonl"
            second = directory / "second.jsonl"

            first_manifest = build_subset(
                source, first, directory / "first.manifest.json", 8, 42
            )
            second_manifest = build_subset(
                source, second, directory / "second.manifest.json", 8, 42
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_manifest["selected_line_numbers"],
                second_manifest["selected_line_numbers"],
            )
            rows = [json.loads(line)["row"] for line in first.read_text().splitlines()]
            self.assertEqual(rows, sorted(rows))
            self.assertEqual(first_manifest["source_sample_count"], 20)
            self.assertEqual(first_manifest["sample_count"], 8)

    def test_rejects_oversized_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source = self._source(directory, count=2)
            with self.assertRaisesRegex(SubsetError, "exceeds source_count"):
                build_subset(
                    source,
                    directory / "subset.jsonl",
                    directory / "manifest.json",
                    sample_count=3,
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()

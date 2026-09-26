from __future__ import annotations

import json
from pathlib import Path

from src.data.materialize_dpo_v1 import MANIFEST, OUTPUT, materialize


def test_frozen_dpo_v1_inventory_materializes_with_integrity(tmp_path: Path) -> None:
    output = tmp_path / "dpo.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = materialize(output_path=output, manifest_path=manifest_path)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 63
    assert len({record["metadata"]["source_sample_id"] for record in records}) == 63
    assert manifest["inventory_counts"] == {
        "grounded_seed_53": 53,
        "phase_5f_verifier_pass_10": 10,
    }
    assert all(record["chosen"]["from"] == "function_call" for record in records)
    assert all(record["rejected"]["from"] == "gpt" for record in records)
    assert all(record["conversations"][-1]["from"] in {"human", "observation"} for record in records)
    assert all(manifest["integrity_checks"].values())


def test_repository_dpo_artifacts_are_current() -> None:
    manifest = materialize()
    persisted = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest == persisted
    assert OUTPUT.exists()

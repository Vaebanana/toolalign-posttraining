# BFCL v4 local evaluation

This pipeline evaluates the final DPO LoRA model on exactly these five
single-turn categories:

- `live_simple` (258)
- `live_multiple` (1,053)
- `parallel` (200)
- `parallel_multiple` (200)
- `irrelevance` (240)

It deliberately excludes multi-turn, memory, web-search, and the other BFCL
collections not covered by this project's training data.

## Environments

Inference runs with the `pt` environment (LLaMA-Factory, Torch, bitsandbytes).
Scoring runs with the separate `bfcl` environment. The runner uses their Python
executables directly, which avoids accidentally using the base Conda Python.

## Full final-model run

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1
```

The defaults load the merged SFT model plus the final DPO adapter:

```text
model:   outputs/models/qwen3_4b_sft_v1_merged
adapter: outputs/train/qwen3_4b_qlora_dpo_v1
```

Raw inference is resumable by BFCL id. Running the same command again skips
completed records. Use `-Overwrite` only when intentionally regenerating every
prediction.

For a foreground run, the pipeline prints a progress line every 10 new records.
For a redirected/background run, follow it with:

```powershell
Get-Content outputs\bfcl\dpo_v1\pipeline.stdout.log -Wait -Tail 30
```

Count durable predictions independently of the log:

```powershell
Get-ChildItem outputs\bfcl\dpo_v1\raw\*.jsonl | ForEach-Object {
    "{0}: {1}" -f $_.Name, @(Get-Content -LiteralPath $_.FullName).Count
}
```

## Individual stages

The three requested stages can also be run separately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1 -Stage Inference
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1 -Stage Convert
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1 -Stage Evaluate
```

The converter copies `raw_prediction` byte-for-byte into BFCL's `result` field.
It validates ids and completeness first. The scoring stage registers a local
Qwen3-FC label in memory, then calls BFCL's official v4 evaluator without
modifying the BFCL checkout.

Official per-category score files and aggregate CSVs are written under:

```text
outputs/bfcl/dpo_v1/score/
```

## Smoke evaluation

One entry per category, including official partial scoring:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1 `
  -RunName smoke `
  -MaxSamplesPerCategory 1 `
  -PartialEval `
  -Overwrite
```

Partial scores are only pipeline checks and must not be reported as full BFCL
results.

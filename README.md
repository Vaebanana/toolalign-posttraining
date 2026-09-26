# ToolAlign Post-Training

[English](README.md) | [简体中文](README.zh-CN.md)

An end-to-end **tool-calling LLM post-training and evaluation pipeline** built around Qwen3-4B-Instruct-2507.

The project is designed around a full experimental loop rather than a single fine-tuning run:

```text
Raw function-calling data
        |
        v
Parse / Audit / Repair / Normalize
        |
        v
Canonical dataset + leakage-aware split
        |
        v
QLoRA SFT
        |
        v
Strict tool-call evaluation + error analysis
        |
        v
On-policy preference mining
        |
        v
Grounded DPO
        |
        v
Internal regression analysis + BFCL v4
```

The main engineering focus is making every stage auditable: deterministic data transforms, strict evaluator semantics, explicit train/test isolation, grounded preference construction, and failure analysis after each training stage.

---

## Project highlights

- Built a canonical function-calling dataset from **xLAM + Hermes** with strict parsing, schema validation, deterministic repair, normalization, and provenance checks.
- Reduced 61,893 raw candidate samples to **59,334 validated canonical samples**.
- Implemented group-aware and tool-aware splitting to prevent query leakage and create a dedicated unseen-tool test set.
- Trained **Qwen3-4B-Instruct-2507** with 4-bit QLoRA under an 8 GB VRAM constraint.
- Built a strict tool-call evaluator that separates response mode, format, tool name, argument exactness, schema validity, and full-call exact match.
- Mined **on-policy SFT failures** and constructed a small, high-confidence **63-pair DPO set** instead of scraping unrelated preference data.
- Added sample-level Base → SFT → DPO transition analysis to measure both fixes and regressions.
- Integrated a local BFCL v4 pipeline for five single-turn tool-use categories.
- Kept weak results visible: DPO produced only a small net gain, while BFCL `irrelevance` exposed over-calling on no-tool cases.

Recorded experiment results are summarized in [results/summary.md](results/summary.md).

---

## 1. Data pipeline

The raw sources used in the project were:

```text
xLAM   60,000 samples
Hermes  1,893 samples
----------------------
Total  61,893 candidates
```

The repository implements parsing, auditing, deterministic repair, normalization, validation, and split verification under `src/data/`.

After validation:

```text
61,893 raw candidates
        |
        v
59,334 canonical usable samples
```

The code also verifies provenance of repaired samples; the recorded pipeline associates **837 samples** with deterministic repair records.

### Canonical split

The final recorded split is:

| Split | Samples |
| --- | ---: |
| train | 50,575 |
| dev_seen | 2,802 |
| test_seen | 2,795 |
| test_unseen_tools | 3,162 |
| **Total** | **59,334** |

The split is not a simple random row split.

It uses:

- normalized user-query grouping to prevent near-duplicate query leakage;
- held-out tool selection for the unseen-tool test set;
- group-aware train/dev/test splitting for the remaining seen-tool pool;
- explicit checks that held-out tools do not leak into train.

The split implementation is in [src/data/split.py](src/data/split.py) and its acceptance checks are in [src/data/check_split.py](src/data/check_split.py).

---

## 2. SFT

SFT-v1 uses a deterministic **10,000-sample subset** of the 50,575-sample train split.

The subset is sampled without replacement with seed 42 and stored with a manifest so the selection can be reproduced exactly.

Training configuration:

```text
Base model:      Qwen/Qwen3-4B-Instruct-2507
Method:          QLoRA
Quantization:    4-bit bitsandbytes
LoRA rank:       8
LoRA targets:    all
Template:        qwen3_nothink
SFT samples:     10,000
Epochs:          1
Learning rate:   1e-4
Batch size:      1
Grad accumulation: 8
```

Config: [configs/sft/qwen3_4b_qlora.yaml](configs/sft/qwen3_4b_qlora.yaml)

Recorded strict full-call results:

| Split | SFT-v1 Full Call Success | Improvement over base |
| --- | ---: | ---: |
| test_seen | **84.51%** | **+7.23 pp** |
| test_unseen_tools | **88.33%** | **+6.45 pp** |

The unseen-tool split is important: it tests whether training improved the general tool-calling behavior rather than only memorizing tool names seen during training.

---

## 3. Strict evaluation

The project does not score tool calling with a single permissive string match.

The evaluator separates:

```text
response mode
format validity
tool-name correctness
argument exactness
schema validity
full-call exact match
no-tool correctness
```

The primary internal metric is `full_call_success_rate`.

Parsing is intentionally strict: malformed tags, prose outside tool calls, Python-literal fallback, type coercion, and silent argument repair are not accepted.

Relevant code:

- [src/eval/parser.py](src/eval/parser.py)
- [src/eval/metrics.py](src/eval/metrics.py)
- [src/eval/evaluate.py](src/eval/evaluate.py)
- [src/eval/inference.py](src/eval/inference.py)

This makes later error mining more meaningful because an error label corresponds to an explicit evaluator rule.

---

## 4. Error analysis and on-policy preference mining

After SFT, the pipeline aligns Base and SFT predictions sample-by-sample and assigns transitions such as:

```text
still_correct
fixed_by_sft
regressed_by_sft
still_wrong
```

The analysis then decomposes failures into concrete categories such as:

- wrong response mode;
- malformed tool-call format;
- wrong / missing / extra tool;
- argument mismatch;
- schema invalidity.

Instead of collecting generic external DPO pairs, the project mines **SFT-v1's own residual failures**.

This is an on-policy design:

```text
prompt
  +
chosen  = grounded canonical tool call
  +
rejected = exact frozen SFT-v1 generation
```

The preference construction code checks that the chosen answer is grounded in the query/tool schema and filters cases where the gold answer is ambiguous, over-specified, default-equivalent, or otherwise unsafe to prefer.

---

## 5. DPO

The final DPO-v1 inventory contains **63 unique preference pairs**.

The materializer verifies that:

- every pair comes from canonical train data;
- DPO samples do not overlap the 10K SFT-v1 subset;
- chosen equals canonical gold;
- rejected equals the frozen SFT-v1 output;
- chosen and rejected differ;
- rejected fails the frozen evaluator;
- the retained preference passes the grounding gate.

The 63 pairs consist of a grounded 53-sample seed inventory plus 10 additional verifier-backed cases.

Training configuration:

```text
Initialization:   merged SFT-v1
Method:           QLoRA DPO
Preference loss:  sigmoid
beta:             0.1
LoRA rank:        8
Epochs:           3
Learning rate:    5e-6
Scheduler:        cosine
Warmup ratio:     0.1
```

Config: [configs/dpo/qwen3_4b_qlora_dpo_v1.yaml](configs/dpo/qwen3_4b_qlora_dpo_v1.yaml)

The recorded Base/SFT/DPO regression analysis found:

```text
DPO fixes:       34
DPO regressions: 32
Net change:      +2
```

This is intentionally reported as a **small and unstable net gain**, not a broad improvement. The result suggests that a 63-pair preference set can correct some targeted residual errors, but is too small and narrow to move overall behavior reliably.

---

## 6. BFCL v4 evaluation

The final DPO adapter is also evaluated with the official BFCL v4 evaluator on five single-turn categories covered by the project:

| BFCL category | Samples | Accuracy |
| --- | ---: | ---: |
| live_simple | 258 | **82.56%** |
| live_multiple | 1,053 | **75.40%** |
| parallel | 200 | **92.50%** |
| parallel_multiple | 200 | **94.50%** |
| irrelevance | 240 | **27.50%** |
| **Micro average** | **1,951** | **74.17%** |

The result has a clear asymmetry:

- structured function selection and parallel tool calling are relatively strong;
- `irrelevance` is much weaker.

The main error-analysis conclusion is therefore not simply "tool calling improved." The model learned tool-call structure and execution behavior better than it learned **whether a tool should be called at all**.

The low irrelevance score exposes an over-calling tendency and a weakness in the no-tool decision boundary.

BFCL integration:

- [docs/bfcl_v4.md](docs/bfcl_v4.md)
- [scripts/run_bfcl_v4.ps1](scripts/run_bfcl_v4.ps1)
- [src/eval/bfcl_inference.py](src/eval/bfcl_inference.py)
- [src/eval/bfcl_convert.py](src/eval/bfcl_convert.py)
- [src/eval/bfcl_evaluate.py](src/eval/bfcl_evaluate.py)

---

## Repository structure

```text
configs/
├── sft/                 # QLoRA SFT configs
├── dpo/                 # merge / smoke / DPO configs
└── analysis/            # frozen adjudication and audit records

data/
└── tools/               # dataset inspection utilities

src/
├── data/                # parse, audit, repair, normalize, split, DPO mining
└── eval/                # inference, strict metrics, error analysis, BFCL adapter

scripts/
└── run_bfcl_v4.ps1

docs/
└── bfcl_v4.md

results/
└── summary.md           # compact committed experiment snapshot

tests/                   # evaluator, mining, DPO and BFCL regression tests
```

Raw datasets, processed datasets, model weights, adapters, checkpoints, and full prediction outputs are intentionally excluded from Git.

---

## Reproducing the main stages

The lightweight data utilities use the dependencies in `requirements.txt`.

Training and local inference require a separate LLaMA-Factory / PyTorch / bitsandbytes environment. BFCL scoring uses the official BFCL environment described in [docs/bfcl_v4.md](docs/bfcl_v4.md).

Representative training commands:

```bash
llamafactory-cli train configs/sft/qwen3_4b_qlora.yaml
llamafactory-cli export configs/dpo/qwen3_4b_sft_v1_merge.yaml
llamafactory-cli train configs/dpo/qwen3_4b_qlora_dpo_v1.yaml
```

BFCL on Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1
```

Because raw datasets and large model artifacts are not committed, this repository should be read as the **reproducible pipeline, configs, audit records, evaluation code, and recorded result summary**, not as a self-contained distribution of the training data or checkpoints.

---

## What the project demonstrates

The useful conclusion of the project is broader than one fine-tuned checkpoint:

1. Tool-calling post-training is highly sensitive to data normalization and evaluator semantics.
2. Seen-tool accuracy alone is insufficient; unseen-tool holdout and external benchmarks expose different failure modes.
3. Preference data should be grounded before DPO; many apparent "model errors" are actually ambiguous or over-specified gold targets.
4. DPO can fix targeted residual errors while simultaneously introducing regressions, so sample-level transition analysis matters.
5. High function-call accuracy can coexist with poor no-tool judgment, as shown by the BFCL irrelevance result.

The project therefore treats **data quality, evaluation design, and error analysis as first-class parts of post-training**, rather than reporting training loss alone.

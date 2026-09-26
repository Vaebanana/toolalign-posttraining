# Experiment Results

This file is a compact snapshot of the final recorded project results. Large raw datasets, model checkpoints, prediction JSONL files, and full evaluation outputs are intentionally excluded from Git.

## Data

| Item | Count |
| --- | ---: |
| xLAM raw samples | 60,000 |
| Hermes raw samples | 1,893 |
| Raw candidates | 61,893 |
| Canonical usable samples | 59,334 |
| train | 50,575 |
| dev_seen | 2,802 |
| test_seen | 2,795 |
| test_unseen_tools | 3,162 |
| deterministic repair-linked samples | 837 |

SFT-v1 uses a deterministic 10,000-sample subset of the train split with seed 42.

## SFT-v1

Model: `Qwen/Qwen3-4B-Instruct-2507`

Method: 4-bit QLoRA, LoRA rank 8, all linear targets, one epoch.

Primary internal metric: strict full-call success.

| Split | SFT-v1 | Improvement over base |
| --- | ---: | ---: |
| test_seen | 84.51% | +7.23 pp |
| test_unseen_tools | 88.33% | +6.45 pp |

## DPO-v1

Preference inventory: 63 unique on-policy pairs.

Construction:

```text
chosen   = grounded canonical tool call
rejected = exact frozen SFT-v1 generation
```

Training: sigmoid DPO, beta 0.1, 3 epochs, QLoRA rank 8.

Sample-level regression analysis:

| Transition caused by DPO | Count |
| --- | ---: |
| Fixed | 34 |
| Regressed | 32 |
| Net | +2 |

Interpretation: DPO corrected some targeted residual errors but produced only a small net gain. The 63-pair set should be treated as a targeted small-scale preference experiment rather than evidence of broad DPO improvement.

## BFCL v4

The final DPO model was evaluated on the five single-turn categories covered by the project.

| Category | Samples | Accuracy |
| --- | ---: | ---: |
| live_simple | 258 | 82.56% |
| live_multiple | 1,053 | 75.40% |
| parallel | 200 | 92.50% |
| parallel_multiple | 200 | 94.50% |
| irrelevance | 240 | 27.50% |
| **Micro average** | **1,951** | **74.17%** |

The main diagnostic result is the gap between strong parallel tool-calling performance and weak `irrelevance` performance. The model is substantially better at producing structured tool calls than at deciding that no tool should be called.

## Reporting caveats

- Internal SFT/DPO metrics and BFCL evaluate different protocols and should not be directly compared as if they were the same accuracy.
- BFCL evaluation intentionally covers only the five single-turn categories listed above; multi-turn, memory, web-search and other collections are outside this project scope.
- The DPO inventory is small by design and includes automated/verifier-assisted grounding plus recorded adjudication evidence; it must not be described as 63 fully human-authored preference pairs.
- Raw generated outputs and checkpoints are omitted from the public repository; this file preserves the final recorded metrics used in the project report and resume.

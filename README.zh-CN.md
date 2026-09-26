# ToolAlign：工具调用大模型后训练与评测 Pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个围绕 **Qwen3-4B-Instruct-2507** 构建的完整 Tool-Calling 后训练项目。

项目重点不是单独完成一次 SFT，而是建立完整实验闭环：

```text
原始 Function Calling 数据
        ↓
解析 / 审计 / 确定性修复 / 规范化
        ↓
Canonical Dataset + 防泄漏数据划分
        ↓
QLoRA SFT
        ↓
严格 Tool-Calling 评测 + Error Analysis
        ↓
从 SFT 真实错误中挖掘 Preference
        ↓
Grounded On-policy DPO
        ↓
回归分析 + BFCL v4 外部评测
```

整个项目强调可审计性：数据变换可复现、评测语义固定、Train/Test 隔离、Preference Grounding，以及每个训练阶段之后的修复/退化分析。

完整实验结果见：[results/summary.md](results/summary.md)。

---

## 项目核心结果

- 对 xLAM + Hermes 做解析、六层审计、确定性修复、规范化与校验。
- 从 61,893 条原始候选样本得到 **59,334 条可用 Canonical 数据**。
- 使用 query group + held-out tools 构造 seen / unseen-tools 防泄漏划分。
- 在 8 GB 显存约束下使用 4-bit QLoRA 微调 **Qwen3-4B-Instruct-2507**。
- 自己实现严格 Tool-Calling Evaluator，区分格式、工具名、参数、Schema、Full Call 和 No-Tool 判断。
- 从 SFT-v1 自身错误中挖掘 on-policy preference，而不是直接拼接外部 DPO 数据。
- 构造 63 条高置信、可 Ground 的 DPO Preference Pair。
- 对 Base → SFT → DPO 做 sample-level fix / regression 分析。
- 接入 BFCL v4 五个 Single-Turn Tool-Use 类别进行外部评测。
- 保留负面结果：DPO 最终只有很小净收益；BFCL `irrelevance` 只有 27.50%，暴露明显的过度工具调用问题。

---

## 1. 数据治理

项目使用：

```text
xLAM   60,000
Hermes  1,893
--------------
总计   61,893 条候选样本
```

`src/data/` 实现了：

```text
Parser
→ Audit
→ Deterministic Repair
→ Normalize
→ Validate
→ Split
```

最终：

```text
61,893 raw candidates
        ↓
59,334 canonical usable samples
```

校验代码还会核对 repair provenance；正式流程中有 **837 条样本**关联到确定性 repair 记录。

### 数据划分

最终 Split：

| Split | 样本数 |
| --- | ---: |
| train | 50,575 |
| dev_seen | 2,802 |
| test_seen | 2,795 |
| test_unseen_tools | 3,162 |
| **Total** | **59,334** |

这里不是普通 Random Split。

实现中同时保证：

- 相同/近似 User Query Group 不跨 Split；
- 专门选择 held-out tools 构造 `test_unseen_tools`；
- held-out tools 不进入 train；
- seen pool 再执行 group-aware train/dev/test 划分；
- 相同 seed 下划分结果可复现。

对应代码：

- [src/data/split.py](src/data/split.py)
- [src/data/check_split.py](src/data/check_split.py)

---

## 2. SFT

SFT-v1 并没有直接训练完整 50,575 条 train，而是从中固定随机抽取 **10,000 条**作为第一版 SFT 子集。

抽样：

```text
without replacement
seed = 42
sample_count = 10,000
```

同时保存 Manifest，保证样本选择可以复现。

训练配置：

```text
Base Model:        Qwen/Qwen3-4B-Instruct-2507
Method:            QLoRA
Quantization:      4-bit bitsandbytes
LoRA Rank:         8
LoRA Target:       all
Template:          qwen3_nothink
SFT Samples:       10,000
Epochs:            1
Learning Rate:     1e-4
Batch Size:        1
Grad Accumulation: 8
```

配置：[configs/sft/qwen3_4b_qlora.yaml](configs/sft/qwen3_4b_qlora.yaml)

严格 Full Call 指标：

| Split | SFT-v1 | 相对 Base |
| --- | ---: | ---: |
| test_seen | **84.51%** | **+7.23 pp** |
| test_unseen_tools | **88.33%** | **+6.45 pp** |

其中 `test_unseen_tools` 用来测试模型是否学到了更一般的 Tool-Calling 行为，而不仅是记住训练阶段出现过的工具。

---

## 3. 严格 Tool-Calling 评测

项目没有只用一个宽松字符串匹配判断 Tool Call 是否正确。

Evaluator 分开计算：

```text
Response Mode
Format Validity
Tool Name
Argument Exact Match
Schema Validity
Full Call Exact Match
No-Tool Correctness
```

内部主指标是：

```text
full_call_success_rate
```

Parser 有意保持严格：

- 不接受 malformed tool tags；
- 不允许 Tool Call 外混入额外 prose；
- 不使用 Python literal fallback；
- 不做 Value Coercion；
- 不在评测阶段偷偷修模型输出。

代码：

- [src/eval/parser.py](src/eval/parser.py)
- [src/eval/metrics.py](src/eval/metrics.py)
- [src/eval/evaluate.py](src/eval/evaluate.py)
- [src/eval/inference.py](src/eval/inference.py)

这样后续 Error Mining 的错误标签才具有明确含义。

---

## 4. Error Analysis 与 On-policy Preference Mining

SFT 后，项目会把 Base 和 SFT Prediction 按 Sample 对齐，并标记：

```text
still_correct
fixed_by_sft
regressed_by_sft
still_wrong
```

然后继续拆分具体错误：

```text
wrong_mode
format_error
wrong_tool
missing_tool
extra_tool
argument_mismatch
schema_invalid
```

DPO 数据不是从外部随便找 chosen/rejected，而是直接从 **SFT-v1 自己仍然做错的样本**中挖掘。

因此属于 on-policy preference：

```text
Prompt
+
Chosen   = Grounded Canonical Tool Call
+
Rejected = SFT-v1 当时真实生成的错误答案
```

但并不是 evaluator 判错就直接放进 DPO。

项目又增加了 Groundability / Semantic Review，用来排除：

- Gold 本身信息不足；
- Gold 过度具体；
- Schema Default 等价；
- 多种表达都合理；
- opaque ID / credential 无法从 Query 推导；
- chosen 并不能被可靠证明优于 rejected。

这部分是整个 DPO Pipeline 比较重要的设计点：**先确认 Preference 本身可信，再拿去做偏好优化。**

---

## 5. DPO

最终 DPO-v1 使用 **63 条唯一 Preference Pair**。

Materializer 会验证：

- 样本来自 canonical train；
- 与 SFT-v1 的 10K 子集无重叠；
- chosen 与 canonical gold 一致；
- rejected 是被冻结的 SFT-v1 原始生成；
- chosen / rejected 不相同；
- rejected 确实被冻结 Evaluator 判错；
- Preference 通过 Groundability Gate。

63 条由：

```text
53 条 grounded seed
+
10 条 verifier-backed 新样本
=
63
```

组成。

训练配置：

```text
Initialization: merged SFT-v1
Method:         QLoRA DPO
Loss:           sigmoid
beta:           0.1
LoRA Rank:      8
Epochs:         3
Learning Rate:  5e-6
Scheduler:      cosine
Warmup Ratio:   0.1
```

配置：[configs/dpo/qwen3_4b_qlora_dpo_v1.yaml](configs/dpo/qwen3_4b_qlora_dpo_v1.yaml)

DPO 回归分析结果：

```text
修复：34
退化：32
净变化：+2
```

因此这里不能总结成“DPO 显著提升”。

更准确的结论是：

> 63 条高置信 Preference 能修复部分定向 residual error，但数据规模太小、覆盖面太窄，整体收益很小，而且存在几乎同量级的 Regression。

---

## 6. BFCL v4

最终 DPO Adapter 使用 BFCL v4 官方 Evaluator，在项目覆盖的 5 个 Single-Turn 类别上评测：

| BFCL Category | 样本数 | Accuracy |
| --- | ---: | ---: |
| live_simple | 258 | **82.56%** |
| live_multiple | 1,053 | **75.40%** |
| parallel | 200 | **92.50%** |
| parallel_multiple | 200 | **94.50%** |
| irrelevance | 240 | **27.50%** |
| **Micro Average** | **1,951** | **74.17%** |

这个结果最重要的不是 74.17% 这个总分，而是不同能力之间非常明显的不均衡：

```text
Parallel Tool Calling        强
Tool Call Structure          较强
When NOT to call a tool      弱
```

尤其：

```text
irrelevance = 27.50%
```

说明模型存在明显 Over-Calling。

因此项目最终的 Error Analysis 结论是：

> 模型对“工具怎么调用”的学习明显好于对“什么时候根本不应该调用工具”的学习。

这也说明单独使用大量正向 Function Calling 样本并不能自动学好 No-Tool Decision Boundary。

BFCL 实现：

- [docs/bfcl_v4.md](docs/bfcl_v4.md)
- [scripts/run_bfcl_v4.ps1](scripts/run_bfcl_v4.ps1)
- [src/eval/bfcl_inference.py](src/eval/bfcl_inference.py)
- [src/eval/bfcl_convert.py](src/eval/bfcl_convert.py)
- [src/eval/bfcl_evaluate.py](src/eval/bfcl_evaluate.py)

---

## 仓库结构

```text
configs/
├── sft/                 # SFT 配置
├── dpo/                 # Merge / Smoke / DPO 配置
└── analysis/            # 人工/辅助审计与 Preference Adjudication

data/
└── tools/               # 原始数据分析工具

src/
├── data/                # Parser / Audit / Repair / Normalize / Split / DPO Mining
└── eval/                # Inference / Metrics / Error Analysis / BFCL

scripts/
└── run_bfcl_v4.ps1

docs/
└── bfcl_v4.md

results/
└── summary.md           # 精简实验结果快照

tests/                   # Data / Evaluator / DPO / BFCL 回归测试
```

以下内容刻意不提交到 Git：

```text
Raw Dataset
Processed Dataset
Model Weights
LoRA Adapters
Merged Models
Checkpoints
完整 Prediction / Evaluation Output
BFCL Raw Output
```

GitHub 只保留能够说明和复现方法的代码、配置、审计记录与关键实验结果。

---

## 运行主要阶段

轻量数据处理依赖：

```bash
pip install -r requirements.txt
```

训练与本地推理依赖单独的 LLaMA-Factory / PyTorch / bitsandbytes 环境。

代表性训练命令：

```bash
llamafactory-cli train configs/sft/qwen3_4b_qlora.yaml
llamafactory-cli export configs/dpo/qwen3_4b_sft_v1_merge.yaml
llamafactory-cli train configs/dpo/qwen3_4b_qlora_dpo_v1.yaml
```

BFCL：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_bfcl_v4.ps1
```

BFCL 环境说明见：

[docs/bfcl_v4.md](docs/bfcl_v4.md)

由于原始数据、模型权重和完整实验输出没有提交，本仓库定位是：

> **可审查的 Pipeline + Config + Evaluation + Audit Evidence + Result Snapshot**

而不是一个包含所有训练数据和 Checkpoint 的自包含模型发布仓库。

---

## 项目最终结论

这个项目最终得到的经验不是简单的“SFT 有效、DPO 有效”。

更准确的是：

1. Tool Calling 后训练首先是数据与评测问题，Canonicalization 和严格 Evaluator 会直接决定你所谓的“正确率”到底代表什么。
2. Seen Test 不够，需要 Unseen-Tool Split 和外部 Benchmark 才能观察泛化。
3. Preference Pair 的质量比数量更关键；很多 evaluator error 并不天然构成可靠 DPO preference。
4. DPO 既会修复也会退化，因此必须做 sample-level regression analysis，而不能只报一个 aggregate metric。
5. Tool Call Accuracy 高，并不意味着 Tool-Use Decision 好；BFCL irrelevance 暴露了明显的 No-Tool Boundary 问题。

因此这个项目把 **数据治理、严格评测、Preference Grounding 和误差分析** 与训练本身放在同等重要的位置。

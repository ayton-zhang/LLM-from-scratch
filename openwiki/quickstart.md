---
type: 快速开始
title: LLM from Scratch 仓库快速开始
description: 九阶段 PyTorch 学习课程的入口，覆盖 Transformer、GPTModern、训练扩展、MoE、SFT、奖励模型、PPO 和 GRPO 的代码、工件与运行约定。
resource: README.md
tags: [pytorch, llm, 课程, 快速开始]
---

# LLM from Scratch

这是一个可执行、面向学习的 PyTorch 课程，不是可安装库或生产服务。九个目录从 attention 原语逐步推进到 tiny GPT、现代推理特性、可扩展预训练、Mixture-of-Experts 和教学化的对齐栈。每个阶段都使用本地导入、脚本和测试；通常应在该阶段自己的目录中运行 `orchestrator.py`。

## 从这里开始

按 `README.md` 创建 Python 3.11 环境：

```bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

随后在目标阶段的目录中执行。例如：

```bash
cd part_1
python orchestrator.py --visualize
```

单独运行该阶段的测试：

```bash
cd part_6
python -m pytest -q
```

工作目录是运行时约定的一部分：模块使用 `from gating import TopKGate` 这类 sibling imports；不支持在仓库根目录收集 `pytest`。完整命令、Docker、工件和前置条件见[测试与运行](operations/testing-and-runs.md)。

## 课程地图

| Part | 目标 | 主要入口 | 产物或依赖 |
|---|---|---|---|
| 1 | 位置编码、因果 attention、MHA、FFN、残差 Transformer block | `part_1/orchestrator.py` | 形状/数学演示和可选 attention 图片 |
| 2 | byte-tokenized tiny GPT、训练、采样和验证 | `part_2/orchestrator.py` | `part_2/runs/min-gpt/model_best.pt` |
| 3 | RMSNorm、RoPE、SwiGLU、GQA、KV cache、sliding-window attention | `part_3/orchestrator.py` | 下游复用的现代模型实现 |
| 4 | BPE、AMP、梯度累积、LR scheduler、checkpoint、日志 | `part_4/orchestrator.py` | BPE tokenizer 和预训练 checkpoint |
| 5 | Top-k MoE 路由与 dense/MoE hybrid FFN | `part_5/orchestrator.py` | 独立组件演示；未接入 Parts 6–9 |
| 6 | 仅对回答计算损失的 supervised fine-tuning | `part_6/orchestrator.py --demo` | SFT policy checkpoint |
| 7 | 成对偏好 reward modeling | `part_7/orchestrator.py --demo` | reward-model checkpoint |
| 8 | 教学化 on-policy PPO RLHF | `part_8/orchestrator.py --demo` | PPO policy checkpoint |
| 9 | 教学化 group-relative policy optimization | `part_9/orchestrator.py --demo` | GRPO policy checkpoint |

[模型与训练架构](architecture/model-and-training.md)解释 Parts 1–5 如何构建模型和训练基础；[对齐工作流](workflows/alignment.md)说明 Parts 6–9 的数据与 checkpoint 链。这两条主线在 Part 3 `GPTModern` 及 Part 4 tokenizer/预训练 checkpoint 处汇合，而[测试与运行](operations/testing-and-runs.md)定义它们的生成、兼容性和运行方式。

## 如何阅读仓库

1. 先读每个阶段的 `orchestrator.py`；它是当前支持的 demo 和检查入口的可靠地图。
2. 先跟踪数据路径，再看模型路径：tokenizer/dataset 或 collator、model、loss、训练循环、checkpoint、sampler/evaluator。
3. 详细中文行内注释和 `notes/explanations/` 是学习辅助；行为以可执行代码为准，因为部分注释来自历史版本或复制而来。
4. 实验时维护跨阶段不变量：模型维度、vocabulary size、tokenizer 文件、padding 约定和 checkpoint 路径必须一致。
5. HEAD 新增的 `StudyVault/` 将上述课程、工件流和开发环境整理为学习笔记；它补充导航，不替代源码和本 wiki 的工程约定。

近期历史表明仓库按教学阶段逐步演进：Parts 6–9 依次形成对齐链，随后提交集中于 SFT label shift/masking、PPO rollout、RollingKV、测试与讲解注释。文档因此区分设计概念与已知的简化或未接通实现。

## Backlog

- **生成模型质量和 benchmark 结果** — 锚点：`part_*/runs/`；未记录运行工件和可复现指标。
- **CI 自动化** — 锚点：`.github/workflows/`；当前工作流目录未纳入已确认的仓库运行行为。
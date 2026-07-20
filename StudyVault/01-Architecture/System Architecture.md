---
module: architecture
path: StudyVault/01-Architecture
keywords: curriculum, transformer, pytorch, architecture
---

# 系统架构

#arch-curriculum #arch-dataflow #pattern-transformer

## 架构模式

按阶段递进的单体教学代码库：每个目录可独立执行，但 Part 3/4/6–9 在代码和 checkpoint 上形成复用关系。

```text
Part 1: 可观察的 attention 数学
       ↓（概念复用，非 import）
Part 2: byte GPT + 训练
       ↓（架构演进）
Part 3: GPTModern
       ↓（直接 import）
Part 4: BPE 预训练 + checkpoint/tokenizer
       ├─→ Part 6: SFT policy
       ├─→ Part 7: reward model
       └─→ Parts 8/9: RLHF

Part 5: 独立的 MoE FFN 实验
```

## 模块边界

| 边界 | 职责 | 关键依赖 |
|---|---|---|
| Parts 1–3 | 模型组件和生成 | PyTorch |
| Part 4 | tokenizer、数据、训练状态 | Part 3 `GPTModern` |
| Part 5 | top-k 路由与专家组合 | 独立，不接入主链 |
| Parts 6–7 | SFT 数据/偏好数据与监督目标 | Part 3/4 资产 |
| Parts 8–9 | rollout 与策略目标 | SFT、reward model、BPE |

## 关键约束

> [!important]
> checkpoint 兼容性不只是词表大小相同：token ID 映射、模型层数/头数/维度、block size 与 tokenizer 路径都必须匹配。

## Related Notes

- [[Learning and Artifact Flow]]
- [[Part 3 — Modern Architecture]]
- [[Part 4 — Training System]]

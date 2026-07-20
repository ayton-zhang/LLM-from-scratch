---
module: architecture
path: StudyVault/01-Architecture
keywords: dataflow, checkpoints, tokenizer, rlhf
---

# 学习与工件流

#arch-dataflow #config-checkpoint #pattern-rlhf

## 数据流

```text
文本 → byte IDs (Part 2) 或 BPE IDs (Part 4)
     → shifted x/y windows → GPT/GPTModern → logits → cross-entropy

instruction + response → masked causal labels (Part 6) → SFT policy
chosen/rejected pairs → encoder + scalar head (Part 7) → reward model
prompts → sampled response → reward + reference log-probs → PPO / GRPO update
```

## 工件与消费者

| 工件 | 生产者 | 消费者 | 不变量 |
|---|---|---|---|
| BPE tokenizer | Part 4 | Parts 6–9 | 同一 token 映射与路径 |
| pretrained `GPTModern` | Part 4 | Part 6 | 配置与 vocab 匹配 |
| SFT policy | Part 6 | Parts 8–9 | 作为初始 policy 和 frozen reference |
| reward model | Part 7 | Parts 8–9 | RM 配置与 tokenizer 匹配 |

## 两条 RL 分支

| 方法 | baseline | value head | reference 约束 |
|---|---|---|---|
| PPO | `returns - old_values` | 使用 | token reward shaping |
| GRPO | 每 prompt 的组内平均 reward | 不使用（但对象仍含 head） | 显式 sampled log-prob 差项 |

> [!warning]
> 这些实现是为教学压缩过的实现。PPO 的 `gamma`/`lambda` 参数没有实现完整 discounted return/GAE；GRPO 的“KL”不是完整分布 KL。

## Related Notes

- [[Part 4 — Training System]]
- [[Part 6 — Supervised Fine-Tuning]]
- [[Part 7 — Reward Modeling]]
- [[Part 8 — PPO RLHF]]
- [[Part 9 — GRPO]]

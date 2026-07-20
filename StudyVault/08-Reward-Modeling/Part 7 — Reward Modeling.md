---
module: part-7
path: part_7
keywords: reward-model, preferences, bradley-terry, ranking
---

# Part 7 — 奖励建模 (★★★)

#module-part-7 #pattern-rlhf #test-unit

## Purpose

在 chosen/rejected 偏好对上训练一个序列→标量 reward model，为 Part 8/9 的 rollout 提供奖励信号。

## Key Files

| 文件 | 角色 |
|---|---|
| `data_prefs.py` | `PrefExample` 与偏好数据加载/回退 |
| `collator_rm.py` | 两边序列格式化、tokenization 和 padding |
| `model_reward.py` | TransformerEncoder、masked mean pool、标量 head |
| `loss_reward.py` | Bradley–Terry 与 margin ranking loss |
| `train_rm.py` / `eval_rm.py` | 训练、pairwise accuracy |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `PairCollator` | class | `(chosen, rejected)` 批次 |
| `RewardModel` | class | token IDs → scalar reward |
| `bradley_terry_loss` | function | 偏好差的 logistic ranking loss |
| `margin_ranking_loss` | function | 强制 reward margin |

## Internal Flow

```text
prompt + chosen/rejected → BPE token pairs → bidirectional TransformerEncoder
                       → mask-aware mean pooling → scalar r+ / r-
                       → ranking loss → RM checkpoint
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | Part 4 BPE、Part 6 formatter | token identity 与文字模板 |
| Used by | Parts 8–9 | 计算 rollout terminal reward |

## Configuration

| 配置 | 用途 | 注意 |
|---|---|---|
| `block_size` | 序列截断 | RM 和 rollout 应匹配 |
| `margin` | ranking 间隔 | 默认 `1.0` |
| padding ID | pooling mask | 与 BPE/rollout 约定一致 |

## Testing

- Run: `cd part_7 && python orchestrator.py`
- Pattern: scalar shape/gradient 与 Bradley–Terry 单调性。

> [!warning]
> pairwise accuracy 只检查 `r_chosen > r_rejected`；不是对奖励质量或安全性的充分评估。

## Related Notes

- [[Part 6 — Supervised Fine-Tuning]]
- [[Part 8 — PPO RLHF]]
- [[Part 9 — GRPO]]
- [[Part 7 Exercises]]

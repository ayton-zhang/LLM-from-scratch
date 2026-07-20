---
module: part-9
path: part_9
keywords: grpo, rlhf, group-baseline, policy-gradient
---

# Part 9 — GRPO (★★★)

#module-part-9 #pattern-rlhf #test-unit

## Purpose

为每个 prompt 采样一组回答，以组内平均 reward 作为 baseline，使用无 value loss 的 PPO-style policy objective 加显式 reference 项。

## Key Files

| 文件 | 角色 |
|---|---|
| `train_grpo.py` | group rollouts、reward、advantages、update |
| `grpo_loss.py` | `ppo_policy_only_losses` 与 diagnostics |
| `policy.py` | 复用 `PolicyWithValue`，但忽略 value head |
| `rollout.py` | 与 Part 8 基本同构的 tokenizer/log-prob 工具 |
| `eval_ppo.py` | copy-forward 的小规模评估脚本 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `PolicyOnlyLossOut` | class | policy-only 损失诊断 |
| `ppo_policy_only_losses` | function | clipped objective、entropy、reference 项 |
| `compute_reward` | function | 生成 response 的 RM 分数 |

## Internal Flow

```text
prompt → k completions → k rewards → reward - per-prompt mean
       → broadcast one trajectory advantage to its response tokens
       → flatten/normalize → clipped policy objective + explicit ref difference
       → updated policy checkpoint
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | Parts 6 / 7 / 4 | SFT policy、RM、BPE tokenizer |
| Mirrors | Part 8 | policy/rollout/evaluation architecture |

## Configuration

| 配置 | 用途 | 注意 |
|---|---|---|
| `group_size` | 每 prompt completion 数 | 决定组内 baseline |
| `clip_ratio` | policy ratio clipping | PPO-style |
| reference term | sampled `new_logp-ref_logp` | 被称作 KL，但不是完整分布 KL |

## Testing

- Run: `cd part_9 && python orchestrator.py`
- Pattern: finite scalar loss 与 diagnostics。

> [!warning]
> advantage 在广播到 token 后再 normalize，因此较长回答对 flattened loss 有更多条目；value head 被实例化但不参与 GRPO loss。

## Related Notes

- [[Part 8 — PPO RLHF]]
- [[Learning and Artifact Flow]]
- [[Part 9 Exercises]]

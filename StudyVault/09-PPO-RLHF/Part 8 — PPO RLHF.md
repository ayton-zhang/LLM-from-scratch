---
module: part-8
path: part_8
keywords: ppo, rlhf, value-head, rollout, kl
---

# Part 8 — PPO RLHF (★★★)

#module-part-8 #pattern-rlhf #test-unit

## Purpose

从 SFT policy 复制 trainable policy 和 frozen reference，采样回答、用 reward model 评分，然后用精简 PPO policy/value objective 更新。

## Key Files

| 文件 | 角色 |
|---|---|
| `policy.py` | `PolicyWithValue`：GPTModern + toy value head |
| `rollout.py` | prompt、采样、shift、log-prob、近似 KL 工具 |
| `train_ppo.py` | rollout、reward、单轮 update |
| `ppo_loss.py` | clipped policy、value、entropy loss |
| `eval_ppo.py` | 小规模 reward 对比 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `PolicyWithValue` | class | logits、value、generation |
| `RLHFTokenizer` | class | 对齐阶段 tokenizer 包装 |
| `gather_logprobs`, `model_logprobs` | function | 选择 response token log-prob |
| `ppo_losses` | function | 输出 policy/value/entropy/total diagnostics |

## Internal Flow

```text
prompts → policy.generate → response IDs → reward model terminal reward
       → response log-probs (policy/reference) + old values
       → KL-shaped immediate rewards → returns - old_values → normalized adv
       → clipped policy loss + MSE value loss - entropy bonus
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | Part 6 checkpoint | policy 与 frozen reference |
| Uses | Part 7 checkpoint | scalar reward |
| Uses | Part 6 formatter / Part 4 BPE | prompt 与 token identity |

## Configuration

| 配置 | 用途 | 实际限制 |
|---|---|---|
| `clip_ratio` | PPO ratio clipping | 默认教学值 |
| `ent_coef` | sampled-token entropy 权重 | 非全分布 entropy |
| `gamma`, `lambda` | 暴露的参数 | 未实现完整 GAE/discounted return |

## Testing

- Run: `cd part_8 && python orchestrator.py`
- Pattern: policy output shape 与标量 clipped objective。

> [!warning]
> 每次 fresh rollout 只做一轮 update；reference divergence 在 advantage 前作为 token reward shaping，而非完整生产 RLHF pipeline。

## Related Notes

- [[Part 7 — Reward Modeling]]
- [[Part 9 — GRPO]]
- [[Part 8 Exercises]]

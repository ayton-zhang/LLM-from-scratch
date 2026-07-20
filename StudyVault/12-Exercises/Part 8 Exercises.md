---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, ppo, rlhf, rollout, part-8
---

# Part 8 — 练习

#practice #onboarding #module-part-8

## Related Modules
- [[Part 8 — PPO RLHF]]

## 练习 1 — Code Reading [trace]
> 从一个 prompt 开始，追踪 PPO 更新的一次完整路径。
> [!answer]- 查看答案
> policy 生成回答，RM 给 terminal reward，rollout 提取 policy/reference log-probs 和 values，形成 advantages，调用 `ppo_losses` 更新。

## 练习 2 — Recall [recall]
> PPO ratio 的公式是什么？
> [!answer]- 查看答案
> `exp(new_logp - old_logp)`；clipping 限制过大的策略变化。

## 练习 3 — Application [config]
> 想减弱策略偏离 reference 的趋势，应调哪里？
> [!answer]- 查看答案
> 检查训练中 reference divergence 的奖励塑形/相关系数，并验证其符号与 rollout token mask。

## 练习 4 — Debugging [debug]
> loss 为 NaN，优先查看什么张量？
> [!answer]- 查看答案
> 检查 selected response token 的 log-probs、ratio、advantages 标准差、returns/values 以及 response mask 是否为空。

## 练习 5 — Analysis [analysis]
> 为什么该实现不是完整 GAE PPO？
> [!answer]- 查看答案
> 虽暴露 `gamma/lambda`，returns 是 immediate shaped rewards，advantages 是 `returns-old_values`，没有折扣回报或 GAE 递推。

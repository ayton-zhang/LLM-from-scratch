---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, reward-model, ranking, part-7
---

# Part 7 — 练习

#practice #onboarding #module-part-7

## Related Modules
- [[Part 7 — Reward Modeling]]

## 练习 1 — Code Reading [trace]
> reward model 怎样把 token 序列变为一个 reward？
> [!answer]- 查看答案
> `RewardModel` 用 bidirectional TransformerEncoder 表示序列，按非 padding token 做 masked mean pooling，再投影为标量。

## 练习 2 — Recall [recall]
> Bradley–Terry loss 偏好什么 reward 关系？
> [!answer]- 查看答案
> 偏好 `r_chosen > r_rejected`；实现为 `mean(softplus(-(r_pos-r_neg)))`。

## 练习 3 — Application [config]
> margin ranking 的 margin 增大有什么含义？
> [!answer]- 查看答案
> 训练会要求 chosen reward 比 rejected reward 高出更大的间隔，可能更强约束也更难满足。

## 练习 4 — Debugging [debug]
> reward 全为相近值，先排查哪两处？
> [!answer]- 查看答案
> 检查 pair collator 是否正确区分 chosen/rejected，以及 padding mask 是否让 pooling 排除了有效 token。

## 练习 5 — Analysis [analysis]
> 为何 pairwise accuracy 不足以证明 RM 可用于安全对齐？
> [!answer]- 查看答案
> 它只在有限偏好对上判断顺序，没有检验分布外、奖励欺骗、校准或人类偏好覆盖。

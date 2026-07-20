---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, grpo, rlhf, group-baseline, part-9
---

# Part 9 — 练习

#practice #onboarding #module-part-9

## Related Modules
- [[Part 9 — GRPO]]

## 练习 1 — Code Reading [trace]
> GRPO 如何为一个 prompt 计算 advantage？
> [!answer]- 查看答案
> 生成 `group_size` 个 completion，逐个 RM 打分，减去本组 reward 平均值，再把该 trajectory advantage 广播给 response tokens。

## 练习 2 — Recall [recall]
> GRPO 是否在 loss 中使用 value head？
> [!answer]- 查看答案
> 不使用；对象仍是 `PolicyWithValue`，但 value head 被忽略，没有 value loss。

## 练习 3 — Application [config]
> `group_size` 从 2 提高到 8 的主要代价和收益？
> [!answer]- 查看答案
> 组内 baseline 估计更丰富，但每 prompt 要生成和打分更多回答，成本明显上升。

## 练习 4 — Debugging [debug]
> 组内 advantages 都为零，先检查什么？
> [!answer]- 查看答案
> 检查同一 prompt 的多次 completion 是否真正不同，以及 RM 是否对它们产生可区分的 rewards。

## 练习 5 — Analysis [analysis]
> 为什么长回答会对当前 flattened loss 有更大影响？
> [!answer]- 查看答案
> trajectory advantage 被广播到每个 response token，随后被扁平化/标准化；长序列贡献更多 token 项。

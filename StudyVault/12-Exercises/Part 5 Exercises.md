---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, moe, routing, part-5
---

# Part 5 — 练习

#practice #onboarding #module-part-5

## Related Modules
- [[Part 5 — Mixture of Experts]]

## 练习 1 — Code Reading [trace]
> token 如何在 `MoE.forward` 中到达所选专家并回到输出位置？
> [!answer]- 查看答案
> gate 产生 top-k IDs/weights；按 expert dispatch token，执行 ExpertMLP，用相应权重 scatter/add 回原 token 位置。

## 练习 2 — Recall [recall]
> 此实现的 selected top-k gate weights 是否重新归一化？
> [!answer]- 查看答案
> 不会；因此所选权重和可以小于 1。

## 练习 3 — Configuration [config]
> 将 `k=1` 改为 `k=2` 直接改变什么？
> [!answer]- 查看答案
> 每 token 会分发给两个专家并加权合并，计算量和负载语义都会变化。

## 练习 4 — Debugging [debug]
> 要确认路由是否塌缩到少数专家，查看什么？
> [!answer]- 查看答案
> 运行 `demo_moe.py` 的路由直方图，并检查 gate 的 top-k 选择和 balance loss 的输入统计。

## 练习 5 — Analysis [analysis]
> 为什么不能据此推断真实分布式 MoE 的吞吐？
> [!answer]- 查看答案
> 没有 capacity、drop policy、expert parallel 或 all-to-all；且 Python loops 重点是可读性。

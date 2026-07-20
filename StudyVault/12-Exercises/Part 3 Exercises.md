---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, rope, kv-cache, part-3
---

# Part 3 — 练习

#practice #onboarding #module-part-3

## Related Modules
- [[Part 3 — Modern Architecture]]

## 练习 1 — Code Reading [trace]
> cached generation 的 prefill 与 decode 分别做什么？
> [!answer]- 查看答案
> prefill 对完整 prompt 建立每层 K/V；decode 每步只输入一个新 token，并把历史 K/V 传回 attention。

## 练习 2 — Recall [recall]
> RMSNorm 与 LayerNorm 的关键差异？
> [!answer]- 查看答案
> RMSNorm 按均方根缩放但不做均值中心化，参数与计算更简单。

## 练习 3 — Application [config]
> 如何启用较少 KV heads 的 attention？
> [!answer]- 查看答案
> 配置兼容的 `n_kv_head` 以启用 GQA；确保 query heads 与 KV heads 的分组关系合法。

## 练习 4 — Debugging [debug]
> 长生成内存没有受 `window + sink` 限制，为什么？
> [!answer]- 查看答案
> `RollingKV` 虽独立受测，`GPTModern.generate()` 主缓存返回路径没有使用它，可能继续累积未裁剪 cache。

## 练习 5 — Analysis [analysis]
> 为什么裁剪 cache 后不能用 cache length 当作 RoPE 的绝对位置？
> [!answer]- 查看答案
> 被丢弃的是中间/旧 token，保留 token 的绝对位置不连续；需要独立追踪真实 position。

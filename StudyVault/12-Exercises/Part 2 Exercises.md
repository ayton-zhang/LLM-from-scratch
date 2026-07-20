---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, gpt, tokenizer, part-2
---

# Part 2 — 练习

#practice #onboarding #module-part-2

## Related Modules
- [[Part 2 — Tiny GPT]]

## 练习 1 — Code Reading [trace]
> `ByteDataset` 如何从 token 序列产生训练 x/y？
> [!answer]- 查看答案
> 随机选择起点，`x` 是长度 T 的窗口，`y` 是从下一 token 开始的同长度窗口；即 `y` 相对 `x` 右移一位。

## 练习 2 — Recall [recall]
> byte tokenizer 的词表大小和覆盖范围是什么？
> [!answer]- 查看答案
> 256 个 ID，对应 UTF-8 bytes `0..255`；它可表达任意字节序列但序列通常较长。

## 练习 3 — Application [config]
> 采样输出太重复，优先调整哪些参数？
> [!answer]- 查看答案
> 提高 temperature，或放宽 top-k/top-p；同时留意随机性提高可能降低连贯性。

## 练习 4 — Debugging [debug]
> checkpoint 加载后 shape mismatch，先核对什么？
> [!answer]- 查看答案
> 核对 `vocab_size`、`block_size`、层数、头数和嵌入维度；优先使用带配置的 `model_best.pt`。

## 练习 5 — Analysis [analysis]
> 为什么 Part 2 的 generation 会促成 KV cache 的需求？
> [!answer]- 查看答案
> 它每生成一个 token 都重新计算整个裁剪上下文的 K/V；cache 可以重用历史 K/V，避免重复计算。

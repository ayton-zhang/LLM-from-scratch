---
type: 架构指南
title: 模型与训练架构
description: 说明 Parts 1–5 如何将 attention 原语演进为 GPTModern、基于 BPE 的训练基础设施、缓存推理，以及独立的稀疏 Mixture-of-Experts 前馈层。
resource: part_3/model_modern.py
tags: [架构, transformer, 预训练, moe]
---

# 模型与训练架构

Parts 1–4 有意从可观察的张量数学逐步构建为可复用的现代语言模型和训练栈；Part 5 将 MoE 作为独立的前馈替换方案探索。形成的 `GPTModern` 和 Part 4 工件为[对齐工作流](../workflows/alignment.md)提供模型与 tokenizer，而其可执行检查和工件路径由[测试与运行](../operations/testing-and-runs.md)维护。

## Part 1：让 attention 可观察

Part 1 以显式操作而非优化为优先。`part_1/multi_head.py` 将 `(B,T,C)` 投影为 Q/K/V，重塑为 `(B,H,T,D)`，应用 scaled causal attention，合并 heads，并返回输出与 attention weights。`part_1/block.py` 以 pre-normalized 残差连接包裹 attention 和 4× GELU FFN。

NumPy demo、形状推演和可选 heatmap 的目的，是在引入训练之前让矩阵运算可检查。代码是独立教学实现；后续模型重新实现架构，并不 import Part 1。

## Part 2：训练第一个 causal LM

首个端到端流程为：

```text
UTF-8 文件
  -> byte IDs (0..255)
  -> 90/10 训练/验证划分
  -> 随机 x 窗口与右移一位的 y 窗口
  -> token + learned position embeddings
  -> pre-norm causal Transformer blocks
  -> LM head 与展平 cross-entropy
  -> 验证、checkpoint、autoregressive sampling
```

`part_2/dataset.py` 实现数据右移；`part_2/model_gpt.py` 包含架构和生成；`part_2/train.py` 管理 AdamW、clipping、可选 AMP/compile、定期评估和 checkpoint。生成会把上下文裁到 `block_size`，并在每个 token 上重算整段窗口，这正是 Part 3 cache 的动机。

`model_best.pt` 是最稳妥的采样工件，因为它带有 `part_2/sample.py` 所需模型配置；最终 checkpoint 格式的信息较少，未必能正确重载非默认架构。

## Part 3：现代模型与缓存推理

`part_3/model_modern.py` 移除 learned positional embeddings，并组合可选的现代组件：

- `rmsnorm.py`：不做均值中心化的 scale normalization。
- `rope_custom.py`：应用于 attention query/key 的 rotary 位置信息。
- `swiglu.py`：门控 FFN activation。
- `attn_modern.py`：支持 grouped-query attention、sliding windows、attention sinks 以及 K/V 输入输出的 causal attention。
- `kv_cache.py`：cache 容器和有界 `RollingKV`；它保留最初 `sink` 与最新 `window` 个位置。

autoregressive generation 先对完整 prompt 做一次 **prefill**，再在每个 **decode** 步仅传入一个新 token 并复用之前的 K/V；这使重复 attention 设置从完整窗口重算变为增量复用。

### Cache 注意事项

单独测试的 `RollingKV` 保证 `length <= sink + window`，但 `GPTModern.generate()` 没有将它接入主 cache 路径。`attn_modern.py` 会为一次 attention 计算裁剪 K/V，却用未裁剪的旧 cache 加新 K/V 来构造返回 cache；长时间生成仍可能让返回 cache 持续增长。此外，一旦丢弃中间旧 token，使用 cache length 作为 RoPE 下一位置并不充分；生产级 streaming 设计需要显式 absolute-position tracking。应将 sliding-window demo 视为教学示例，而不是生产内存上界保证。

## Part 4：扩展训练循环

`part_4/train.py` 从 Part 3 import `GPTModern`，并在训练中固定启用 RMSNorm、RoPE 和 SwiGLU。其流程为：

```text
文本文件 -> 训练/加载 BPE -> token 化语料 -> 重叠的右移窗口
-> DataLoader -> GPTModern -> cross-entropy
-> AMP-scaled 和/或 accumulated gradients -> AdamW
-> warmup + cosine scheduler -> logger -> checkpoints
```

周边模块各自承担职责：

- `tokenizer_bpe.py`：训练并持久化 Hugging Face BPE tokenizer。
- `dataset_bpe.py`：创建右移训练窗口；尽管名称带有 “streaming”，仍会在内存中读取并 token 化整个文件。
- `amp_accum.py`：处理 autocast/scaling 与 gradient accumulation。
- `lr_scheduler.py`：实现 warmup 加 cosine decay。
- `checkpointing.py`：保存 model、optimizer、scheduler、scaler、step 和模型配置。
- `logger.py`：支持 TensorBoard 和可选 WandB；WandB 动态 import，但未固定在 `requirements.txt`。

resume 和下游使用依赖原 tokenizer 路径与兼容的 model/vocabulary dimensions。移动 run 目录可能破坏 checkpoint 所记录的 tokenizer 位置。训练停止取决于 `--steps`，scheduler horizon 则由 dataset/epoch 计算；非典型组合可使二者分歧。

## Part 5：作为 FFN 替代方案的稀疏 experts

Part 5 是独立组件学习，不属于 Part 4–9 checkpoint 链。`part_5/gating.py` 计算 softmax expert probabilities，并为每个 token 选取 top-k experts。`part_5/moe.py` 通过显式 Python loops 把 token 分发到独立 MLP，再用选中的 gate weights 合成输出。

关键语义：

- 选出的 top-k weights 不会重新归一化，因此其和可能小于一。
- 平衡项是 `E * sum(importance * load)`，其中 importance 使用 soft probabilities，load 使用 top-1 assignment frequency。
- 不含 capacity factor、dropped-token policy、expert parallelism 或 all-to-all communication。
- `block_hybrid.py` 的 `HybridFFN` 返回 `alpha * Dense(x) + (1-alpha) * MoE(x)`，默认 `alpha=0.5`；即使端点值，两条分支仍都会执行。

该实现让 routing mechanics 易于观察，但不应被解读为面向性能的 MoE 系统。`part_5/README.md` 明确将生产 expert parallelism 列为范围之外。

## 变更指引

- 修改 attention 时保持 `(B,H,T,D)` 契约和 causal masking；运行 Part 1 与 Part 3 测试。
- 修改 model configuration 时，必须同步 checkpoint 创建、加载、采样和每个下游阶段。
- tokenizer 改动会同时影响 Parts 4 与 6–9；即使 vocabulary size 相同，也不能把 token IDs 不一致的 checkpoint 混用。
- cache 改动需要比现有测试更强的验证：比较 cached/uncached logits 与生成，并覆盖超过 configured window 的序列。
- MoE 改动应测试 routing gradients 和每个 expert 的利用率，而非只测输出形状和“存在某些 gradient”。
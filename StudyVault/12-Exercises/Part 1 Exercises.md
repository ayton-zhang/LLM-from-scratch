---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, attention, transformer, part-1
---

# Part 1 — 练习

#practice #onboarding #module-part-1

## Related Modules
- [[Part 1 — Transformer Foundations]]

## 练习 1 — Code Reading [trace]
> 从输入 `x (B,T,C)` 开始，追踪多头注意力如何得到输出。
> [!answer]- 查看答案
> `MultiHeadSelfAttention.forward` 投影 Q/K/V，重排为 `(B,H,T,D)`，做 scaled masked attention，拼接 heads 后 output projection。

## 练习 2 — Recall [recall]
> 为什么 causal mask 必须阻止位置 t 看见未来位置？
> [!answer]- 查看答案
> 否则训练时会泄露 label token，推理时又没有未来 token，造成 train/inference mismatch。

## 练习 3 — Configuration [config]
> 将 256 维表示改为 8 个头时，每头维度是什么？应检查什么？
> [!answer]- 查看答案
> `head_dim=32`；检查 `d_model % n_head == 0`，并按 `sqrt(32)` 缩放注意力分数。

## 练习 4 — Debugging [debug]
> 若 attention weights 在未来位置非零，先看哪里？
> [!answer]- 查看答案
> 检查 `attn_mask.causal_mask` 的上三角方向，以及 softmax 前是否将被屏蔽位置设为极小值。

## 练习 5 — Extension [extend]
> 如何比较 learned 和 sinusoidal 位置编码？
> [!answer]- 查看答案
> 在相同输入/维度下替换 `pos_encoding.py` 的实现，比较参数量、长度外推行为和下游输出；增加 shape test。

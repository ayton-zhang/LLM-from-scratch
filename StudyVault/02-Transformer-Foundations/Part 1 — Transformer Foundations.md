---
module: part-1
path: part_1
keywords: attention, positional-encoding, transformer, pytorch
---

# Part 1 — Transformer 基础 (★★★)

#module-part-1 #pattern-transformer #test-unit

## Purpose

用透明的张量操作构建位置编码、因果注意力、多头注意力、FFN 与 pre-norm Transformer Block；目标是看懂形状和数学，而非训练模型。

## Key Files

| 文件 | 角色 |
|---|---|
| `attn_mask.py` | `causal_mask(T)`，禁止看未来 token |
| `single_head.py` / `multi_head.py` | 单头及多头自注意力 |
| `pos_encoding.py` | learned 与 sinusoidal 位置编码 |
| `ffn.py` / `block.py` | MLP 与残差 Transformer block |
| `orchestrator.py` | 测试与可视化编排 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `causal_mask` | function | 创建上三角屏蔽矩阵 |
| `SingleHeadSelfAttention` | class | 计算一个注意力头 |
| `MultiHeadSelfAttention` | class | 分头、注意力、拼接、投影 |
| `TransformerBlock` | class | attention + FFN 的残差层 |

## Internal Flow

```text
x (B,T,C) → Q,K,V projection → reshape (B,H,T,D)
          → scaled QKᵀ + causal mask → softmax → V
          → merge heads → output projection
          → residual + FFN → block output
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | PyTorch / NumPy / matplotlib | 张量、手算、可视化 |
| Used by | 学习者与 Part 1 demos | 概念复用，后续不直接 import |

## Configuration

| 参数 | 含义 | 默认 |
|---|---|---|
| `d_model`, `n_head` | 模型维度与头数 | 调用方指定 |
| `dropout` | 注意力/FFN dropout | `0.0` |

## Testing

- Run: `cd part_1 && python orchestrator.py`
- Pattern: `test_causal_mask.py` 验证 mask；`test_attn_math.py` 对照 NumPy 手算。

> [!tip]
> 每个 head 的维度必须满足 `d_model % n_head == 0`；缩放项使用 `sqrt(head_dim)`，不是 `sqrt(d_model)`。

## Related Notes

- [[Part 2 — Tiny GPT]]
- [[System Architecture]]
- [[Part 1 Exercises]]

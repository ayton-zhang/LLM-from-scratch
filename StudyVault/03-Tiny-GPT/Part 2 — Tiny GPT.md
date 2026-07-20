---
module: part-2
path: part_2
keywords: gpt, byte-tokenizer, training, sampling
---

# Part 2 — Tiny GPT (★★★)

#module-part-2 #pattern-transformer #test-unit

## Purpose

把 Part 1 的结构接入第一个端到端因果语言模型：从 UTF-8 bytes 构造 next-token 数据、训练 GPT、验证 loss 并采样。

## Key Files

| 文件 | 角色 |
|---|---|
| `tokenizer.py` | `ByteTokenizer`，ID 范围 0–255 |
| `dataset.py` | 90/10 切分与 shifted windows |
| `model_gpt.py` | `GPT`、causal attention、block、generation |
| `train.py` | AdamW、clip、可选 AMP/compile、checkpoint |
| `sample.py` / `eval_loss.py` | 采样与验证 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `ByteTokenizer` | class | UTF-8 文本和 byte IDs 互转 |
| `ByteDataset` | class | 提供 `(x, y)` 的一 token 偏移批次 |
| `GPT.forward` | method | logits 与可选交叉熵 |
| `GPT.generate` | method | temperature/top-k/top-p 自回归生成 |

## Internal Flow

```text
text → UTF-8 bytes → x[t:t+T], y[t+1:t+T+1]
     → token embedding + learned position embedding
     → N × pre-norm block → LM head → flattened cross-entropy
     → checkpoint / sampled bytes
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | PyTorch | 模型与 AdamW |
| Used by | Part 2 scripts | Part 3 会重新实现模型，不直接依赖 |

## Configuration

| 配置 | 用途 | 注意 |
|---|---|---|
| `block_size` | 上下文窗口 | 生成时会裁剪输入 |
| `n_layer/n_head/n_embd` | GPT 形状 | 与 checkpoint 一致 |
| `top_k/top_p/temperature` | 采样分布 | 可联合使用 |

## Testing

- Run: `cd part_2 && python orchestrator.py`
- Pattern: tokenizer round-trip 与精确的 `x/y` shift 对齐。

> [!warning]
> `model_best.pt` 比最终 checkpoint 更适合采样，因为它携带创建模型的配置；生成每 token 会重算完整窗口。

## Related Notes

- [[Part 1 — Transformer Foundations]]
- [[Part 3 — Modern Architecture]]
- [[Part 2 Exercises]]

---
module: part-3
path: part_3
keywords: rmsnorm, rope, swiglu, kv-cache, gqa
---

# Part 3 — 现代架构 (★★★)

#module-part-3 #pattern-transformer #arch-dataflow #test-unit

## Purpose

实现可被后续阶段复用的 `GPTModern`：RMSNorm、RoPE、SwiGLU、GQA、KV cache、滑动窗口与 attention sink。

## Key Files

| 文件 | 角色 |
|---|---|
| `model_modern.py` | `GPTModern`，forward、cached/no-cache generation |
| `attn_modern.py` | 现代因果注意力与可选 GQA/window |
| `rmsnorm.py`, `rope_custom.py`, `swiglu.py` | 三个核心组件 |
| `kv_cache.py` | `KVCache` 与受限的 `RollingKV` |
| `block_modern.py` | 组件装配成 block |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `GPTModern` | class | 后续预训练/SFT 所用 LM |
| `RMSNorm` / `SwiGLU` | class | normalization 与门控 FFN |
| `RoPECache`, `apply_rope_single` | class/function | 旋转位置编码 |
| `KVCache`, `RollingKV` | class | K/V 重用或受限缓冲 |

## Internal Flow

```text
prefill prompt → per-layer K/V cache
decode one token → RoPE(Q,K) → attention over cached K/V → logits
               ↘ next K/V cache ↗
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Used by | Part 4 | `train.py` import `GPTModern` |
| Used by | Part 6 | SFT train/sample import `GPTModern` |
| Uses | PyTorch | attention 与缓存张量 |

## Configuration

| 配置 | 用途 | 风险 |
|---|---|---|
| `n_kv_head` | GQA 的 KV 头数 | 必须与 query heads 兼容 |
| `window`, `sink` | 局部 attention | 不等于端到端 cache 有界 |
| `start_pos` | RoPE 位置 | 流式裁剪时需绝对位置 |

## Testing

- Run: `cd part_3 && python orchestrator.py --skip-demo`
- Pattern: RMSNorm、RoPE 和独立 `RollingKV` 的形状/数值测试。

> [!warning]
> `RollingKV` 受测且有界，但 `GPTModern.generate()` 主 cache 路径没有接入它；长生成时返回 cache 仍可能增长，且裁剪后不能仅用 cache 长度推断 RoPE 位置。

## Related Notes

- [[Part 2 — Tiny GPT]]
- [[Part 4 — Training System]]
- [[Part 3 Exercises]]

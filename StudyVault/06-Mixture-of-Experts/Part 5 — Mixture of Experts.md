---
module: part-5
path: part_5
keywords: moe, routing, experts, load-balancing
---

# Part 5 — Mixture of Experts (★★)

#module-part-5 #pattern-moe #test-unit

## Purpose

独立演示 top-k 路由、专家 MLP 和 dense/MoE 混合 FFN；它不接入 Parts 6–9 checkpoint 链。

## Key Files

| 文件 | 角色 |
|---|---|
| `gating.py` | `TopKGate`，softmax 和 token 的 top-k expert 选择 |
| `experts.py` | 每个 expert 的 MLP |
| `moe.py` | token dispatch、expert 执行、加权组合与 balance loss |
| `block_hybrid.py` | `HybridFFN` 混合 dense/MoE 输出 |
| `demo_moe.py` | 路由直方图演示 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `TopKGate` | class | 选择每 token 的 experts 与权重 |
| `ExpertMLP` | class | 单专家前馈网络 |
| `MoE` | class | 稀疏 dispatch/combine |
| `HybridFFN` | class | `alpha*Dense + (1-alpha)*MoE` |

## Internal Flow

```text
x (B,T,C) → flatten tokens → gate softmax → top-k IDs/weights
           → per-expert token dispatch → ExpertMLP → scatter/add weighted outputs
           → reshape (B,T,C) + balance loss
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | PyTorch | routing、索引、MLP |
| Used by | Part 5 demo/tests | 无下游 checkpoint 消费者 |

## Configuration

| 参数 | 含义 | 当前语义 |
|---|---|---|
| `n_expert`, `k` | 专家数、每 token 选择数 | selected weights 不重新归一化 |
| `alpha` | dense/MoE 混合比 | 两分支总会执行 |
| `mult`, `swiglu` | expert MLP 形状 | 教学优先 |

## Testing

- Run: `cd part_5 && python orchestrator.py --no-demo`
- Pattern: gate shapes、MoE gradient/output、hybrid blending。

> [!warning]
> 没有 capacity factor、dropped-token policy、expert parallel 或 all-to-all 通信；不要把它当作性能型生产 MoE。

## Related Notes

- [[System Architecture]]
- [[Part 4 — Training System]]
- [[Part 5 Exercises]]

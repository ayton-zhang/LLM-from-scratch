---
module: part-6
path: part_6
keywords: sft, instruction, masking, curriculum, evaluation
---

# Part 6 — 监督微调 (★★★)

#module-part-6 #pattern-rlhf #config-checkpoint #test-unit

## Purpose

将 Part 4 预训练模型在 instruction→response 数据上继续训练；仅对 response 预测计算 loss，并输出 SFT policy checkpoint。

## Key Files

| 文件 | 角色 |
|---|---|
| `dataset_sft.py` / `formatters.py` | Alpaca 风格项与稳定 prompt template |
| `collator_sft.py` | tokenization、shifted labels、prompt masking、padding |
| `curriculum.py` | `LengthCurriculum` |
| `train_sft.py` / `sample_sft.py` | 训练与指令生成 |
| `evaluate.py` | exact match 与 token F1 |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `format_example`, `format_prompt_only` | function | 全量/仅 prompt 格式化 |
| `SFTCollator` | class | 生成 causal LM batch |
| `LengthCurriculum` | class | 按长度安排样本 |
| `exact_match`, `token_f1` | function | 简单文本评估 |

## Internal Flow

```text
instruction/input/output → formatted prompt + response → BPE IDs
→ input IDs + shifted labels → prompt labels = -100 → GPTModern loss
→ SFT checkpoint → policy/reference for PPO and GRPO
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | Part 3 `GPTModern`, Part 4 BPE/checkpoint | path injection 后 import |
| Used by | Parts 8–9 | policy 与 prompt format |

## Configuration

| 配置 | 作用 | 不变量 |
|---|---|---|
| `block_size` | 截断长度 | 要与模型一致 |
| padding ID | pad token | 代码假定为 `2` |
| prompt mask | `-100` 忽略 loss | mask 到 `n_prompt - 1`，保留首个 response token 的预测标签 |

## Testing

- Run: `cd part_6 && python -m pytest -q`
- Pattern: template marker 和 label masking；orchestrator 内当前两条测试命令被注释，需直接运行 pytest。

## Related Notes

- [[Part 4 — Training System]]
- [[Part 7 — Reward Modeling]]
- [[Part 6 Exercises]]

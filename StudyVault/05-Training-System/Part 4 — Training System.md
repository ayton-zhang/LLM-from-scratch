---
module: part-4
path: part_4
keywords: bpe, amp, checkpointing, scheduler, training
---

# Part 4 — 训练系统 (★★★)

#module-part-4 #config-checkpoint #test-unit

## Purpose

把 `GPTModern` 接到 BPE tokenizer、DataLoader、AMP/梯度累积、warmup+cosine 学习率、日志和可恢复 checkpoint，产出对齐主链的起点。

## Key Files

| 文件 | 角色 |
|---|---|
| `tokenizer_bpe.py` | `BPETokenizer` 的训练、保存、加载 |
| `dataset_bpe.py` | 全文 tokenization 与 shifted windows |
| `train.py` | 主训练循环和模型配置 |
| `amp_accum.py` | `AmpGrad` 包装 autocast/scaler/accumulation |
| `lr_scheduler.py` | `WarmupCosineLR` |
| `checkpointing.py` | 模型、optim、scheduler、scaler、config 原子保存 |
| `logger.py` | TensorBoard、可选 WandB |

## Public Interface

| 导出 | 类型 | 作用 |
|---|---|---|
| `BPETokenizer` | class | BPE ID 编解码与磁盘生命周期 |
| `TextBPEBuffer` / `make_loader` | class/function | 训练批次 |
| `AmpGrad` | class | accumulation 和 AMP step |
| `WarmupCosineLR` | class | warmup 后 cosine decay |
| `save_checkpoint` / `load_checkpoint` | function | 完整训练状态 |

## Internal Flow

```text
corpus → train/load BPE → full-corpus IDs → shifted windows → DataLoader
      → GPTModern → cross-entropy → AmpGrad/AdamW → scheduler/logger
      → checkpoint(model + optimizer + scheduler + scaler + config)
```

## Dependencies

| 方向 | 模块 | 方式 |
|---|---|---|
| Uses | `part_3/model_modern.py` | 直接 import `GPTModern` |
| Used by | Parts 6–9 | tokenizer、pretrained checkpoint |
| Uses | tokenizers, TensorBoard | BPE 与日志；WandB 未被 requirements 固定 |

## Configuration

| 配置 | 作用 | 注意 |
|---|---|---|
| `--steps` | 训练停止条件 | 可能与 scheduler horizon 不同 |
| `--accum`, AMP | 有效 batch / 显存 | optimizer step 才更新 |
| output tokenizer path | checkpoint 元数据 | 移动 run 目录会导致加载失败 |

## Testing

- Run: `cd part_4 && python orchestrator.py --no-demo`
- Pattern: BPE save/load、scheduler 进度、checkpoint shape smoke tests。

> [!important]
> 下游不能只比较 vocab size；必须复用同一 tokenizer 文件与模型配置。

## Related Notes

- [[Part 3 — Modern Architecture]]
- [[Learning and Artifact Flow]]
- [[Part 4 Exercises]]

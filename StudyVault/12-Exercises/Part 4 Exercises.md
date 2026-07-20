---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, bpe, checkpointing, part-4
---

# Part 4 — 练习

#practice #onboarding #module-part-4

## Related Modules
- [[Part 4 — Training System]]

## 练习 1 — Code Reading [trace]
> 从原始文本到 checkpoint，列出 Part 4 的主数据流。
> [!answer]- 查看答案
> BPE train/load → 全文 IDs → shifted windows/DataLoader → GPTModern/loss → AMP/AdamW/scheduler/logger → 保存完整训练状态。

## 练习 2 — Recall [recall]
> 完整 checkpoint 除模型权重外还保存什么？
> [!answer]- 查看答案
> optimizer、scheduler、AMP scaler、step 与 model config，供恢复训练和兼容性验证。

## 练习 3 — Configuration [config]
> 如何在显存不足时增加有效 batch size？
> [!answer]- 查看答案
> 使用 `AmpGrad` 的 gradient accumulation，并在支持时使用 AMP；注意只在累积完成后 optimizer step。

## 练习 4 — Debugging [debug]
> 移动 `runs/` 后 tokenizer 加载失败，先在哪里修复？
> [!answer]- 查看答案
> 检查 checkpoint 记录的 tokenizer 路径和 `tokenizer/` 是否仍存在；不要静默新建未训练 tokenizer。

## 练习 5 — Analysis [analysis]
> 为什么 `--steps` 与 scheduler horizon 不一致会令人困惑？
> [!answer]- 查看答案
> 停止条件与学习率曲线长度来自不同计算；非默认 epoch/数据量组合可能在衰减计划中途停止或超出。

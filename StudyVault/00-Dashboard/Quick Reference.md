---
module: dashboard
path: StudyVault/00-Dashboard
keywords: commands, environment, tests, checkpoints
---

# 快速参考

#dashboard #config-environment #test-unit

## 环境 → [[Development Environment]]

```bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

## 常用命令 → [[Part 1 — Transformer Foundations]]

| 目的 | 命令 |
|---|---|
| Part 1 检查 | `cd part_1 && python orchestrator.py` |
| Part 3 只跑测试 | `cd part_3 && python orchestrator.py --skip-demo` |
| Part 4 只跑测试 | `cd part_4 && python orchestrator.py --no-demo` |
| 单个阶段全部测试 | `cd part_5 && python -m pytest -q` |
| Docker 进入环境 | `docker compose build && docker compose up -d && docker compose exec llm-lab bash` |

## 对齐工件链 → [[Learning and Artifact Flow]]

```text
part_4/runs/part4-demo/model_last.pt + tokenizer/
  -> part_6/runs/sft-demo/model_last.pt
  -> part_7/runs/rm-demo/model_last.pt
  -> part_8/runs/ppo-demo/model_last.pt  或  part_9/runs/grpo-demo/model_last.pt
```

## 重要位置

| 路径 | 用途 | 说明 |
|---|---|---|
| `part_1/` | 注意力基础 | 形状和可视化 |
| `part_3/model_modern.py` | 可复用现代模型 | 被 Part 4、6 使用 |
| `part_4/tokenizer_bpe.py` | BPE tokenizer | 对齐阶段需保持同一份 tokenizer |
| `part_6/formatters.py` | SFT 提示模板 | Parts 7–9 间接复用 |
| `part_*/tests/` | 组件验证 | 不等于端到端质量评估 |

## 调试优先级

| 症状 | 先检查 | → 笔记 |
|---|---|---|
| `ImportError` | 是否在 `part_N/` 内运行 | [[Development Environment]] |
| checkpoint shape mismatch | 模型维度、vocab、tokenizer 是否一致 | [[Learning and Artifact Flow]] |
| 对齐 demo 找不到工件 | 是否先执行 Part 4、6、7 demo | [[Part 6 — Supervised Fine-Tuning]] |
| 长生成显存增长 | Part 3 cache 长度 | [[Part 3 — Modern Architecture]] |

## Related Notes

- [[LLM from Scratch — Onboarding Map]]
- [[System Architecture]]

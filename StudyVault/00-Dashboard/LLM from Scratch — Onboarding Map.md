---
module: dashboard
path: StudyVault/00-Dashboard
keywords: curriculum, pytorch, transformer, alignment, onboarding
---

# LLM from Scratch — 学习地图

#dashboard #onboarding #arch-curriculum

## 架构总览

这是一个九阶段的可执行 PyTorch 课程，不是可安装的服务或库。每个 `part_N/` 以本地导入为前提，应从其自身目录运行 `orchestrator.py`。

→ [[System Architecture]]  
→ [[Learning and Artifact Flow]]

## 模块地图

| 模块 | 目标 | 入口 | 笔记 |
|---|---|---|---|
| Part 1 | 从张量数学理解 Transformer | `part_1/orchestrator.py` | [[Part 1 — Transformer Foundations]] |
| Part 2 | 训练第一个 byte-level GPT | `part_2/orchestrator.py` | [[Part 2 — Tiny GPT]] |
| Part 3 | 加入现代 LLM 组件与缓存 | `part_3/orchestrator.py` | [[Part 3 — Modern Architecture]] |
| Part 4 | 扩展训练工程 | `part_4/orchestrator.py` | [[Part 4 — Training System]] |
| Part 5 | 理解稀疏 MoE | `part_5/orchestrator.py` | [[Part 5 — Mixture of Experts]] |
| Part 6 | 指令监督微调 | `part_6/orchestrator.py` | [[Part 6 — Supervised Fine-Tuning]] |
| Part 7 | 偏好奖励建模 | `part_7/orchestrator.py` | [[Part 7 — Reward Modeling]] |
| Part 8 | PPO 式 RLHF | `part_8/orchestrator.py` | [[Part 8 — PPO RLHF]] |
| Part 9 | GRPO 式 RLHF | `part_9/orchestrator.py` | [[Part 9 — GRPO]] |

## API / 命令面

没有 HTTP API；公共交互面是各部分的 Python 类、脚本与测试命令。→ [[Quick Reference]]、[[Development Environment]]

## 标签索引

| 标签 | 含义 | 规则 |
|---|---|---|
| `#arch-curriculum` / `#arch-dataflow` | 课程架构、端到端数据流 | 架构笔记使用 |
| `#module-part-1` … `#module-part-9` | 单个课程模块 | 模块笔记及其练习必须共用 |
| `#pattern-transformer` / `#pattern-rlhf` / `#pattern-moe` | 关键实现模式 | 与对应模块标签共用 |
| `#config-environment` / `#config-checkpoint` | 环境或配置约束 | 仅工程说明使用 |
| `#test-unit` | 单元测试与验证 | 与模块标签共用 |
| `#practice` / `#onboarding` | 主动练习与阅读路径 | 仅练习、导航笔记使用 |

## 推荐学习路径

1. [[System Architecture]]：先建立课程和依赖关系。
2. [[Part 1 — Transformer Foundations]] → [[Part 2 — Tiny GPT]]：先学计算，再学训练。
3. [[Part 3 — Modern Architecture]] → [[Part 4 — Training System]]：理解可复用模型与工件。
4. [[Part 5 — Mixture of Experts]]：独立专题，不进入对齐主链。
5. [[Part 6 — Supervised Fine-Tuning]] → [[Part 7 — Reward Modeling]] → [[Part 8 — PPO RLHF]] 或 [[Part 9 — GRPO]]。
6. 每章阅读后做对应的 [[Part 1 Exercises]] 至 [[Part 9 Exercises]]。

## 学习提醒

> [!warning]
> 不要从仓库根目录直接运行 pytest。各阶段以 sibling imports 为运行约定；进入 `part_N/` 后再运行测试。

## Related Notes

- [[Quick Reference]]
- [[Development Environment]]
- [[Learning and Artifact Flow]]

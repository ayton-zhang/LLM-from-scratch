---
type: 运行手册
title: 测试与运行
description: 用于配置 Python 或 Docker 环境、运行各课程阶段、定位生成 checkpoint，并理解测试覆盖与运行边界的实用手册。
resource: requirements.txt
tags: [测试, 运维, checkpoints, docker]
---

# 测试与运行

本手册运行并验证[模型与训练架构](../architecture/model-and-training.md)和分阶段的[对齐工作流](../workflows/alignment.md)。仓库没有根 package configuration；本地工作目录是运行时契约的一部分。

## 环境

使用 Python 3.11，并执行 `pip install -r requirements.txt`。重要固定依赖包括 PyTorch 2.8、pytest、NumPy、Hugging Face `tokenizers`/`datasets` 和 TensorBoard。小型测试可不使用 CUDA，但训练与生成在 GPU 上更可行。WandB logging 还需单独安装 `wandb`，因为它不在 `requirements.txt`。

Docker 提供替代开发环境：

```bash
docker compose build
docker compose up -d
docker compose exec llm-lab bash
```

`Dockerfile` 构建位于 `/app` 的 Python 3.11 image；`docker-compose.yml` 将仓库 bind-mount 到该处、请求全部 NVIDIA GPUs，并无限 sleep 以供交互使用。直接 `docker run` 默认执行 `python part_1/demo_mha_shapes.py`，而 Compose 覆盖该命令。

## 分阶段检查

从表中指定目录运行命令。

| Part | 测试/检查 | Demo 行为 |
|---|---|---|
| 1 | `python orchestrator.py` | 增加 `--visualize` 可在 `part_1/out/` 写入 attention PNGs |
| 2 | `python orchestrator.py` | 始终训练、采样和评估；启用 AMP 与 `torch.compile` |
| 3 | `python orchestrator.py --skip-demo` | 不传 `--skip-demo` 时，还会以 window/sink 配置生成 200 tokens |
| 4 | `python orchestrator.py --no-demo` | 不传 `--no-demo` 时，训练小型 BPE model 并采样 |
| 5 | `python orchestrator.py --no-demo` | demo 默认运行，除非禁用 |
| 6 | `python -m pytest -q` | `python orchestrator.py` 默认运行 demo；`--no-demo` 跳过 demo，且当前不执行已注释的测试调用 |
| 7 | `python orchestrator.py` | 增加 `--demo`；需要 Part 4 tokenizer |
| 8 | `python orchestrator.py` | 增加 `--demo`；需要 Parts 4、6 和 7 工件 |
| 9 | `python orchestrator.py` | 增加 `--demo`；需要 Parts 4、6 和 7 工件 |

运行某一阶段的完整测试目录：

```bash
cd part_5
python -m pytest -q
```

orchestrator 通常会以正确 `cwd` 在 fail-fast subprocess 中运行选定测试和 demos。避免在仓库根目录运行 `pytest`：测试将 sibling modules 作为 top-level names import，未经 import-path 调整的 collection 会失败。Part 6 是当前例外：测试命令已在其 orchestrator 中注释，必须显式执行 `python -m pytest -q`。

## 工件链

默认 demonstrations 期望下列路径：

```text
part_2/runs/min-gpt/model_best.pt
part_4/runs/part4-demo/
  model_last.pt
  tokenizer/
part_6/runs/sft-demo/model_last.pt
part_7/runs/rm-demo/model_last.pt
part_8/runs/ppo-demo/model_last.pt
part_9/runs/grpo-demo/model_last.pt
```

要运行完整对齐链，先执行 Part 4 demo，再执行 Parts 6 和 7 demos，最后执行 Part 8 或 Part 9。Part 6 模型维度显式匹配 Part 4 smoke model（`n_layer=2`、`n_head=2`、`n_embd=128`）。除非同时匹配 CLI architecture arguments 和 tokenizer，否则不要替换为任意 checkpoint。

生成的 `runs/`、logs、tokenizers 和 checkpoints 是运行工件，不保证已提交。Part 4 会在 checkpoint 旁记录 tokenizer path；移动 run tree 可能使该路径失效。

## 测试实际证明的内容

- **Part 1：**attention/mask shapes 与 finite outputs。
- **Part 2：**byte-tokenizer 行为和精确的 next-token dataset shift。
- **Part 3：**RMSNorm、RoPE shape/value 行为，以及独立 `RollingKV` 的有界 shape。
- **Part 4：**BPE save/load lifecycle、scheduler bounds/progression 与 checkpoint shape smoke behavior。
- **Part 5：**gate shapes、MoE integration、hybrid block output 和最小 gradient flow。
- **Part 6：**formatter 和 masked labels 的基本存在性；需要显式运行 pytest，因为 orchestrator 当前不调度这些测试。
- **Part 7：**reward forward shape 和 Bradley–Terry monotonicity。
- **Part 8：**policy forward shape 和 scalar PPO loss。
- **Part 9：**scalar/finite GRPO loss 与返回 diagnostics。

这些是组件测试，不是模型质量测试。它们不证明跨 Part checkpoint compatibility、tokenizer identity、cached-vs-uncached equivalence、精确 SFT boundary masking、完整 router/expert gradients、PPO rollout correctness 或 GRPO group-baseline construction。

## 故障排查与护栏

- **Import errors：**确认 shell 位于 `part_N/`；不要在不评估所有 orchestrator 的情况下全局“修复” imports。
- **缺少 alignment checkpoint：**运行工件链中列出的上游 demo。
- **Checkpoint size mismatch：**使用创建它的 architecture 与 vocabulary。优先使用 Part 2 的 `model_best.pt`、Part 4 完整 checkpoint 等自描述工件。
- **Tokenizer load failure：**确认原始保存的 tokenizer 目录和 `tokenizer.json`；不要静默创建未训练 BPE tokenizer 作为替代。
- **CPU 速度慢或 compile 问题：**Part 2 orchestrator 会启用 AMP 和 compile；诊断 CPU 环境时，用保守 flags 直接调用 `train.py`。
- **长生成内存：**不要假定 Part 3 已测试的 `RollingKV` 会约束 `GPTModern.generate()` 使用的 cache；显式检查 cache length。
- **Scheduler 异常：**Part 4 的 scheduler horizon 与 `--steps` 停止逻辑来自不同计算；测试非默认 epoch/dataset 组合。

## 修改阶段前

1. 从该阶段目录运行测试。
2. 确认上游输入和下游 checkpoint consumers。
3. 修改 source code，而不是生成的运行工件。
4. 重跑组件测试和最小相关 orchestrator demo。
5. 改动 tokenizer、checkpoint、padding 或 model config 时，手动 smoke-test 紧邻下游阶段，因为现有测试未覆盖这些边界。
---
module: devops
path: .
keywords: python, docker, pytest, environment, commands
---

# 开发环境与运行约定

#config-environment #test-unit

## Purpose

记录本课程可复现的环境、目录工作约定、测试方式与生成工件边界。

## Key Files

| 文件 | 作用 |
|---|---|
| `requirements.txt` | Python 依赖，目标 Python 3.11 |
| `Dockerfile` | Python 3.11 容器环境 |
| `docker-compose.yml` | 挂载仓库并请求 NVIDIA GPU |
| `part_N/orchestrator.py` | 每阶段的正确 cwd 编排入口 |
| `.github/workflows/openwiki-update.yml` | OpenWiki 定时更新 |

## Public Interface

| 命令 | 用途 |
|---|---|
| `pip install -r requirements.txt` | 安装课程依赖 |
| `cd part_N && python orchestrator.py` | 跑该阶段建议检查 |
| `cd part_N && python -m pytest -q` | 收集该阶段测试 |

## Internal Flow

```text
shell cwd = part_N → local sibling imports resolve
                  → orchestrator subprocesses run tests/demos
                  → runs/, logs/, tokenizer/, checkpoints are generated artifacts
```

## Dependencies

| 方向 | 模块 / 服务 | 方式 |
|---|---|---|
| Uses | Python 3.11, PyTorch, pytest | 主开发栈 |
| Optional | CUDA, Docker, TensorBoard, WandB | 速度/可视化；WandB 未固定依赖 |

## Configuration

| 配置 | 目的 | 默认/风险 |
|---|---|---|
| working directory | sibling imports | 必须是对应 `part_N/` |
| run directories | 保存 checkpoint/tokenizer | 默认不保证已提交 |
| GPU | 加速训练 | 小测试可 CPU，训练较慢 |

## Testing

- 根目录 pytest 不被支持；不要通过全局 import path “修复”而破坏模块假设。
- 在改变 tokenizer、checkpoint、padding、模型配置时，先跑本阶段测试，再手动 smoke-test 下游。

## Related Notes

- [[Quick Reference]]
- [[Learning and Artifact Flow]]

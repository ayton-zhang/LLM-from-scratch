# ==========================================
# Orchestrator：Part 4 测试运行器 + 烟雾训练演示
# ==========================================
# 本文件的职责：
#   1. 一键运行 Part 4 的所有单元测试
#   2. 可选运行一个"烟雾训练"——用极小模型在小数据集上快速训练几步，
#      验证训练循环、AMP 混合精度、梯度累积、checkpoint、采样等端到端正常
#
# Part 4 相比 Part 3 的升级：从"推理"走向"训练"。
# Part 3 构建了现代模型架构（RMSNorm + RoPE + SwiGLU + KV Cache），
# Part 4 补齐训练基础设施：
#   - BPE 分词器（替代字节级 tokenizer）
#   - 流式数据集 + batch 拼接
#   - Warmup + 余弦衰减学习率调度
#   - AMP 混合精度训练（加速 + 省显存）
#   - 梯度累积（小 batch 模拟大 batch）
#   - 检查点保存/恢复
#   - 日志记录（TensorBoard / WandB）
#
# 用法：
#   cd part_4
#   python orchestrator.py            # 跑测试 + 烟雾训练 + 采样（默认）
#   python orchestrator.py --no-demo  # 只跑测试

# Repository layout (Part 4)
#
#   part_4/
#     orchestrator.py             # run unit tests + optional smoke train & sample
#     tokenizer_bpe.py            # 4.1 BPE tokenization (train/save/load)
#     dataset_bpe.py              # streaming dataset + batching & label shift
#     lr_scheduler.py             # 4.3 Warmup + cosine decay scheduler
#     amp_accum.py                # 4.2 AMP (autocast+GradScaler) + grad accumulation helpers
#     checkpointing.py            # 4.4 save/resume (model/opt/scaler/scheduler/tokenizer)
#     logger.py                   # 4.5 logging backends (wandb / tensorboard / noop)
#     train.py                    # core training loop (no Trainer API)
#     sample.py                   # load checkpoint & generate text
#     tests/
#       test_tokenizer_bpe.py
#       test_scheduler.py
#       test_resume_shapes.py
#
# Run from inside `part_4/`:
#   cd part_4
#   python orchestrator.py --demo      # tiny smoke run on ../tiny.txt
#   pytest -q
#   tensorboard --logdir=runs/part4-demo

import argparse, pathlib, subprocess, sys, shlex

# ROOT：本文件所在目录的绝对路径，即 part_4/。
# 所有子进程的 cwd 都设为此路径，确保从任何位置调用都能正确找到模块。
# 语法：pathlib.Path(__file__).resolve().parent 是获取"当前脚本所在目录"的现代写法。
ROOT = pathlib.Path(__file__).resolve().parent

# ==========================================
# 工具函数：在子进程中运行命令
# ==========================================
def run(cmd: str):
    """在 ROOT 目录下运行一条 shell 命令，失败时退出整个程序。

    用 subprocess 而非直接 import 运行测试/train：
    - 隔离：每个命令独立进程，全局状态不互相污染
    - 真实性：等价于用户手动敲命令，能发现 PATH/依赖等环境问题
    - 灵活性：train.py 和 sample.py 是独立脚本，必须用子进程
    """
    print(f"\n>>> {cmd}")
    # shlex.split：按 shell 语法拆分命令字符串，正确处理引号和转义。
    # cwd=ROOT：工作目录设为 part_4/，确保 import 路径正确。
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    # sys.exit(n)：透传子进程退出码。n=0 正常，n≠0 异常。
    # 这样外部 CI/脚本能正确判断成功或失败。
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 主流程
# ==========================================
# 语法：`if __name__ == "__main__":` Python 的标准入口守卫。
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # --no-demo：不加则默认跑烟雾训练 + 采样，加了则跳过。
    # action="store_false" + default=True：传了为 False，没传为 True。
    p.add_argument("--no-demo", action="store_false", dest="demo", default=True,
                    help="skip the smoke train+sample demo (demo runs by default)")
    args = p.parse_args()

    # ─── 第一步：运行单元测试 ───
    # 三个测试文件分别验证：
    #   test_tokenizer_bpe.py → BPE 分词器的训练/编码/解码/保存/加载
    #   test_scheduler.py     → Warmup + 余弦衰减学习率调度曲线正确
    #   test_resume_shapes.py → 检查点保存/恢复后模型形状一致
    #
    # 语法：sys.executable 返回当前 Python 解释器的绝对路径。
    # 子进程的 PATH 可能与当前终端不同（如 virtualenv 环境），
    # 用 sys.executable 保证找到正确的 Python，避免 "python: command not found"。
    # 1) unit tests
    run(f"{sys.executable} -m pytest -q tests/test_tokenizer_bpe.py")
    run(f"{sys.executable} -m pytest -q tests/test_scheduler.py")
    run(f"{sys.executable} -m pytest -q tests/test_resume_shapes.py")

    # ─── 第二步：可选烟雾训练 + 采样 ───
    # 2) optional demo (quick overfit on tiny file)
    if args.demo:
        # ─── 烟雾训练 ───
        # 用 ../part_2/tiny.txt 做数据（极小的纯文本文件），
        # BPE 分词器（vocab_size=8000），极小模型（2 层/2 头/128 维），
        # 只训练 300 步/1 epoch（够快，只验证训练循环能跑通）。
        #
        # 关键参数：
        #   --bpe                : 用 BPE 分词器替代字节级 tokenizer
        #   --vocab_size 8000    : BPE 词表大小（字节级默认 256）
        #   --mixed_precision    : 启用 AMP 混合精度（FP16 前向，FP32 权重）
        #   --grad_accum_steps 2 : 梯度累积 2 步再更新，模拟 batch_size=32 的效果
        #   --log tensorboard    : 训练日志写入 TensorBoard（可用 tensorboard --logdir=... 查看）
        #
        # 语法：`\` 换行符只是视觉分行，Python 会把括号内的多行自动拼接。
        run(f"{sys.executable} train.py --data ../part_2/tiny.txt --out runs/part4-demo --bpe --vocab_size 8000 --epochs 1 --steps 300 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard")

        # ─── 生成采样 ───
        # 加载刚训练的模型 checkpoint，生成 100 个 token 的续写文本。
        # --ckpt model_last.pt：训练最后一步自动保存的检查点
        # （checkpointing.py 每 step_interval 步或训练结束时保存）。
        # --prompt 'Generate a short story'：给定的起始文本，模型续写。
        run(f"{sys.executable} sample.py --ckpt runs/part4-demo/model_last.pt --tokens 100 --prompt 'Generate a short story'")

    print("\nPart 4 checks complete. ✅")

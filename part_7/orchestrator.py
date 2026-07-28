# ==========================================
# Part 7 自动化测试与 Demo 协调调度器 (Orchestrator)
# 职责：依次运行单元测试，并在指定标志位开启时完成极小奖励模型 (RM) 的训练与评估流程。
# ==========================================

# Repository layout (Part 7)
#
#   part_7/
#     orchestrator.py           # run unit tests + optional tiny RM demo
#     data_prefs.py             # 7.1 HF preference loader (+tiny fallback)
#     collator_rm.py            # pairwise tokenization → (pos, neg) tensors
#     model_reward.py           # 7.2 reward model (Transformer encoder → scalar)
#     loss_reward.py            # 7.3 Bradley–Terry & margin-ranking losses
#     train_rm.py               # minimal one‑GPU training on tiny slice
#     eval_rm.py                # 7.4 sanity checks & simple accuracy on val
#     tests/
#       test_bt_loss.py
#       test_reward_forward.py
#
# Run from inside `part_7/`:
#   cd part_7
#   python orchestrator.py --demo
#   pytest -q

import argparse, pathlib, subprocess, sys, shlex

# ─── 路径定位 ───
# 语法：pathlib.Path(__file__).resolve().parent
#   __file__ 表示当前脚本文件的路径；
#   .resolve() 将其解析为绝对路径（消除软链接和相对路径符号 ..）；
#   .parent 获取所在目录。
# 这样保证无论从哪个目录下执行 `python part_7/orchestrator.py`，ROOT 都能精确定位到 part_7/ 目录。
ROOT = pathlib.Path(__file__).resolve().parent

# ==========================================
# 辅助函数：命令行子进程执行与状态校验
# ==========================================

def run(cmd: str):
    # 打印即将执行的命令提示（类似于 shell 中的 set -x 输出）
    print(f"\n>>> {cmd}")

    # 语法：shlex.split(cmd) 按照 Shell 的解析规则将字符串安全拆分为参数列表
    # 例如把 "python train_rm.py --steps 300" 拆分为 ['python', 'train_rm.py', '--steps', '300']
    # 相比字符串直接 .split(' ')，shlex 能妥善处理被双引号包围的单参数等复杂情况。
    # 语法：cwd=ROOT 设定子进程的工作目录为 part_7/，保证脚本引用的相对路径保持一致。
    args = shlex.split(cmd)
    if args and args[0] in ("python", "python3"):
        args[0] = sys.executable
    res = subprocess.run(args, cwd=ROOT)

    # 检查子进程退出码（returncode）:
    # returncode == 0 表示成功执行；非 0 表示发生错误或测试失败（例如 pytest 判定未通过）。
    # 一旦报错立刻调用 sys.exit 中断调度器，防止故障隐蔽传播到下一步。
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 主流程控制入口
# ==========================================

if __name__ == "__main__":
    # 语法：argparse 模块用于构建标准命令行接口 (CLI)
    p = argparse.ArgumentParser()
    # 默认 demo=True：方便直接运行完整 RM 流程，无需手动传 --demo。如需只跑测试，显式传入 --no-demo
    p.add_argument("--demo", action="store_true", default=True, help="tiny reward‑model demo")
    p.add_argument("--no-demo", action="store_false", dest="demo", help="skip the demo, run tests only")
    args = p.parse_args()

    # ─── 阶段 1：运行核心单元测试 ───
    # 验证 Bradley-Terry 偏好损失函数以及奖励模型前向传播张量形状与标量评分逻辑
    run(f"{sys.executable} -m pytest -q tests/test_bt_loss.py")
    run(f"{sys.executable} -m pytest -q tests/test_reward_forward.py")

    # ─── 阶段 2：极小规模训练与效果验证 (Demo) ───
    # 默认开启 demo 验证（通过 args.demo 控制，可通过 --no-demo 关闭）
    if args.demo:
        # 1. 训练微型 RM：采用 2 层 Transformer 编码器、128 隐藏层维度、Bradley-Terry 损失，训练 300 步
        run(f"{sys.executable} train_rm.py --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --loss bt --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # 2. 评估训练集前 8 条偏好对的拟合准确率（Sanity check 检查模型是否能够正常收敛）
        run(f"{sys.executable} eval_rm.py --ckpt runs/rm-demo/model_last.pt --split train[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # 3. 评估测试集前 8 条偏好对的泛化表现
        run(f"{sys.executable} eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 7 checks complete. ✅")
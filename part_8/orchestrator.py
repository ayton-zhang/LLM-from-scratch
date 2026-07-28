# ==========================================
# Part 8 自动化测试与 Demo 协调调度器 (Orchestrator)
# 职责：依次运行 PPO 核心单元测试，并控制微型 PPO RLHF 训练与评估流程的启动。
# ==========================================

# Repository layout (Part 8 — RLHF with PPO)
#
#   part_8/
#     orchestrator.py          # run unit tests + optional tiny PPO demo
#     policy.py                # policy = SFT LM + value head (toy head on logits)
#     rollout.py               # prompt formatting, sampling, logprobs/KL utilities
#     ppo_loss.py              # PPO clipped objective + value + entropy + KL penalty
#     train_ppo.py             # single‑GPU RLHF loop (tiny, on‑policy)
#     eval_ppo.py              # compare reward vs. reference on a small set
#     tests/
#       test_ppo_loss.py
#       test_policy_forward.py
#
# Run from inside `part_8/`:
#   cd part_8
#   python orchestrator.py --demo 
#   pytest -q

# ==========================================
# 导入模块：编排 PPO 训练、测试与评估流程
# ==========================================
# argparse    → 解析命令行参数（--demo / --no-demo）
# pathlib     → 面向对象的路径操作，精确定位当前脚本所在目录
# subprocess  → 在 Python 中创建子进程执行外部命令（如 pytest、train_ppo.py）
# sys         → 用于在子进程异常退出时及时终止调度器
import argparse, pathlib, shlex, subprocess, sys

# ─── 路径定位 ───
# 语法：pathlib.Path(__file__).resolve().parent
#   __file__ 表示当前脚本路径；.resolve() 解析软链接与相对路径；.parent 获取父目录。
# 保证无论从哪个工作目录运行命令，ROOT 都能精确指向 part_8/ 目录。
ROOT = pathlib.Path(__file__).resolve().parent

# ==========================================
# 辅助函数：命令行子进程执行与状态校验
# ==========================================
def run(cmd: str):
    # 打印即将执行的命令提示，方便定位调试
    print(f"\n>>> {cmd}")

    # 语法：shlex.split(cmd) 按 shell 语法拆分命令字符串；如果以 python 开头则使用 sys.executable
    args = shlex.split(cmd)
    if args and args[0] in ("python", "python3"):
        args[0] = sys.executable
    res = subprocess.run(args, cwd=ROOT)

    # 语法：res.returncode 非 0 表示子进程执行报错（如单元测试未通过），此时立即退出，防止错误隐蔽扩散
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 主流程控制入口
# ==========================================
if __name__ == "__main__":
    # ─── 命令行参数解析 ───
    p = argparse.ArgumentParser()
    # 默认 demo=True：方便直接运行完整 PPO 流程；如果只想运行单元测试，显式传入 --no-demo
    p.add_argument("--demo", action="store_true", default=True, help="tiny PPO demo")
    p.add_argument("--no-demo", action="store_false", dest="demo", help="skip the PPO demo, run tests only")
    args = p.parse_args()

    # ─── 阶段 1：运行 PPO 核心单元测试 ───
    # 验证 PPO 剪切损失（Clipped Loss）、Value Loss、Entropy 及 Policy 网络前向传播张量形状
    run("python -m pytest -q tests/test_ppo_loss.py")
    run("python -m pytest -q tests/test_policy_forward.py")

    # ─── 阶段 2：运行极小规模 PPO 训练与效果评估 (Demo) ───
    # 注意：运行 PPO Demo 依赖 Part 6 (SFT) 和 Part 7 (RM) 的检查点权重
    if args.demo:
        # 保留历史步数的配置对比（已注释）：
        # 10 步 / 50 步配置仅用于快速验证代码通道连通性
        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 10 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 50 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # 1. 训练 PPO Policy：加载 Part 6 的 SFT 策略模型与 Part 7 的 Reward 模型，训练 100 步
        run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # 2. 评估 PPO Policy：对比 RLHF 优化后的 Policy 与参考模型的奖励得分情况
        run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 8 checks complete. ✅")
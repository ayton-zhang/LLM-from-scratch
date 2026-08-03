# ==========================================
# Part 9 编排脚本：一键跑单元测试 + 可选微型 GRPO 演示
# ==========================================
# 职责：作为 Part 9 的"总控台"，按顺序执行两件事：
#   1. 运行 grpo_loss 的单元测试（校验损失计算正确性）
#   2. 可选（--demo）：跑一遍完整的 GRPO 训练 + 评估流水线
#
# 设计动机：把"测试 + 训练 + 评估"封装成一个命令，避免手动敲一长串
# 相对路径参数出错。也是每章代码的"验收入口"。
# ==========================================

import argparse, pathlib, shlex, subprocess, sys
# ROOT：本文件所在目录（part_9/），后续所有子进程都以它为工作目录
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    """执行一条 shell 命令，失败则退出整个脚本。

    语法：subprocess.run(args, cwd=ROOT) 启动子进程执行命令；
          cwd=ROOT 确保子进程在 part_9 目录下运行（相对路径才找得到文件）。
    """
    print(f"\n>>> {cmd}")
    # 语法：shlex.split(cmd) 把命令字符串拆成参数列表（正确处理引号，比 str.split() 安全）
    args = shlex.split(cmd)
    # 把 "python"/"python3" 替换为当前解释器路径（sys.executable），
    # 保证子进程使用与当前相同的 Python 环境（避免虚拟环境不一致）
    if args and args[0] in ("python", "python3"):
        args[0] = sys.executable
    res = subprocess.run(args, cwd=ROOT)
    # 子进程返回码非 0 表示失败——直接退出，不再继续跑后面的步骤（失败要尽早暴露）
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 入口：解析参数并依次执行测试 / 演示
# ==========================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # --demo：可选开关，开启后额外执行完整的 GRPO 训练 + 评估流水线
    p.add_argument("--demo", action="store_true", help="tiny GRPO demo")
    args = p.parse_args()

    # 1) 单元测试：校验 grpo_loss 的各项损失计算是否符合预期
    #    语法：python -m pytest -q 以 pytest 方式运行测试文件，-q 静默模式只输出概要
    run("python -m pytest -q tests/test_grpo_loss.py")

    # 2) 可选演示（需要 Part 6 的 SFT checkpoint 与 Part 7 的 RM checkpoint）
    #    --group_size 4  ：每个 prompt 采样 4 个回答组成一个"组"（GRPO 的核心结构）
    #    --batch_prompts 4：每步选 4 个不同的 prompt → 每步共 4×4=16 条轨迹
    #    --resp_len 128   ：回答最长 128 个 token（比默认 64 长，生成更完整）
    #    --bpe_dir        ：指定 Part 4 训练好的 BPE 词表目录（保证词表一致）
    if args.demo:
        run("python train_grpo.py --group_size 4 --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 200 --batch_prompts 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # 训练完成后评估 GRPO 模型，与 SFT 参考模型对比平均奖励
        run("python eval_ppo.py --policy_ckpt runs/grpo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 9 checks complete. ✅")

# ==========================================
# Part 3 编排器：测试运行 + 生成演示
# ==========================================
# 这个脚本是 Part 3 的"总控台"——一键运行所有单元测试，
# 并可选地启动一个轻量级文本生成演示，验证各组件的协同工作。
#
# 用法：
#   cd part_3
#   python orchestrator.py            # 只跑测试
#   python orchestrator.py --demo     # 跑测试 + 生成演示
#   pytest -q                         # 也可以手动跑测试
#
# 测试覆盖的模块：
#   test_rmsnorm.py     → 3.1 RMSNorm（均方根归一化）
#   test_rope_apply.py  → 3.2 RoPE（旋转位置编码）
#   test_kvcache_shapes.py → 3.4/3.6 KV Cache（键值缓存 + 滚动缓冲区）

# 语法：argparse 是 Python 标准库的命令行参数解析器，比手写 sys.argv 更安全、更易读。
# pathlib.Path 是面向对象的文件路径操作库，比 os.path 更现代、更易读。
import argparse, pathlib, subprocess, sys, shlex

# ROOT 指向 part_3/ 目录本身（orchestrator.py 所在的目录）。
# 语法：Path(__file__).resolve() 返回当前脚本的绝对路径，
# .parent 取所在目录，确保无论从哪里执行脚本都能正确定位文件。
ROOT = pathlib.Path(__file__).resolve().parent

# ─── 工具函数：安全地执行命令 ───
def run(cmd: str):
    """在 part_3/ 目录下执行给定的 shell 命令，失败时立即终止整个脚本。

    语法：subprocess.run(...) 启动一个子进程执行命令。
    - shlex.split(cmd) 的作用是把命令字符串按 shell 语法切分成列表，
      例如 "python -m pytest -q" → ["python", "-m", "pytest", "-q"]，
      比手写 .split() 更安全（能正确处理引号和转义字符）。
    - cwd=ROOT 确保命令在 part_3/ 目录下执行，无论用户从哪个目录调用的脚本。
    - 如果命令执行失败（returncode != 0），sys.exit() 立即终止，
      避免在"前置步骤已失败"的情况下继续跑后续命令，造成混乱的错误输出。
    """
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)


# ─── 主入口：解析参数 → 跑测试 → （可选）演示 ───
if __name__ == "__main__":
    # 语法：ArgumentParser 创建一个参数解析器，add_argument 注册可接受的参数。
    # action="store_true" 表示 --demo 是一个布尔开关，指定时 args.demo = True，否则 False。
    # 与之对比：不加 action="store_true" 则默认需要后面跟一个值（如 --epochs 10）。
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny generation demo")
    args = p.parse_args()

    # ─── 阶段 1：跑全部单元测试 ───
    # tests/ 目录下包含 8 个测试文件，按学习目的分为两组：
    #
    #   【形状检查（原 Part 3 自带）】
    #     test_rmsnorm.py           → RMSNorm 输出形状不变
    #     test_rope_apply.py        → RoPE 旋转后形状不变 + 值有变化
    #     test_kvcache_shapes.py    → RollingKV 缓存长度不超 sink+window
    #
    #   【数值正确性 + 集成（本次 debug 学习补充）】
    #     test_rmsnorm_correctness.py → RMS≈1、weight 缩放、梯度流通
    #     test_rope_correctness.py    → RoPE 位置 0 恒等、相对位置影响点积、注意力梯度
    #     test_kvcache_correctness.py → KVCache 值保留、sink 不变、window 最新
    #     test_training_flow.py       → 完整训练前向 + loss + backward
    #     test_generate_flow.py       → KV cache vs 无缓存一致性、prefill/decode 状态
    #
    # python -m pytest tests/ 一次性跑全部，-v 显示每个测试名。
    # 用户可在感兴趣的函数中设断点，通过 debug 跟踪具体组件的计算流程。
    run(f"{sys.executable} -m pytest -q tests/test_training_flow.py")
    run(f"{sys.executable} -m pytest -q tests/test_generate_flow.py")

    # ─── 阶段 2：生成演示（可选）───
    # 只在用户指定 --demo 时才执行。
    # --tokens 200 表示生成 200 个新 token，足够展示一段时间内模型的输出质量，
    # 又不会太耗时（几秒内完成，适合快速验证改动）。
    # 对比：训练用的完整脚本可能跑几千 token 或直至 EOS，这里只需要"看一眼效果"。
    if args.demo:
        run(f"{sys.executable} demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    # ─── 完成提示 ───
    # 如果任何一步失败，sys.exit() 会在此之前终止脚本，所以能打印到这里 = 全部通过。
    print("\nPart 3 checks complete. ✅")

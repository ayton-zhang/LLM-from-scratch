# ==========================================
# Part 5 测试编排器：Mixture-of-Experts 的自动化验证
# ==========================================
#
# 这个脚本是整个 Part 5 的"一键检查"入口：
#   1. 依次运行三个单元测试文件，验证门控、MoE 前向传播、混合 Block 的正确性
#   2. 可选（--demo）：跑一遍 demo_moe.py，输出路由直方图等可视化信息
#
# 设计思路：
#   单元测试覆盖边界情况（形状、dtype、组合方式），而 demo 则是集成验证，
#   确保所有组件拼在一起能正常运行。两者结合 = 快速反馈 + 端到端信心。
#
# 运行方式（在 part_5/ 目录下）：
#   python orchestrator.py          # 只跑测试
#   python orchestrator.py --demo   # 跑测试 + MoE demo

# ==========================================
# 导入区：标准库工具
# ==========================================

# argparse：构建命令行接口，让脚本接受 --demo 等参数
import argparse
# pathlib：面向对象的文件路径操作，比 os.path 更直观
import pathlib
# subprocess：在 Python 中启动子进程，这里用于调用 pytest 和 demo 脚本
import subprocess
# sys：获取命令行参数、控制进程退出码
import sys
# shlex：按 shell 语法拆分命令字符串为参数列表（正确处理引号、转义等）
import shlex

# ==========================================
# 全局配置
# ==========================================

# 语法：pathlib.Path(__file__).resolve().parent
#   __file__          → 当前脚本的路径（可能是相对路径）
#   .resolve()        → 解析为绝对路径
#   .parent           → 取父目录，即 part_5/
# 效果：无论从哪里执行此脚本，ROOT 始终指向 part_5/ 目录
ROOT = pathlib.Path(__file__).resolve().parent


# ==========================================
# run()：子进程执行器
# ==========================================
def run(cmd: str):
    """在 ROOT（part_5/）目录下执行一条 shell 命令，失败则立即终止整个流程。

    参数:
        cmd : 要执行的命令字符串，如 "python -m pytest -q tests/test_gate_shapes.py"
    """
    # 打印命令，方便用户看到当前正在执行哪一步
    print(f"\n>>> {cmd}")

    # 语法：shlex.split(cmd)
    #   把命令字符串拆成列表，例如 "python -m pytest -q" → ["python", "-m", "pytest", "-q"]
    #   比手写 .split() 更安全：它正确处理引号包裹的参数（如 "test name with spaces"）
    # 语法：subprocess.run(args, cwd=ROOT)
    #   cwd 参数指定子进程的工作目录——无论脚本从哪里被调用，命令都在 part_5/ 下执行
    #   subprocess.run 会阻塞等待子进程结束，返回一个 CompletedProcess 对象
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)

    # 语法：res.returncode
    #   子进程的退出码，0 = 成功，非 0 = 失败
    # 如果任一测试失败，立即以相同的退出码终止整个编排脚本——"快速失败"原则
    if res.returncode != 0:
        sys.exit(res.returncode)


# ==========================================
# 主入口：测试编排流程
# ==========================================
if __name__ == "__main__":
    # ─── 步骤 0：解析命令行参数 ───
    # argparse.ArgumentParser：创建一个命令行参数解析器
    p = argparse.ArgumentParser()

    # 语法：p.add_argument("--demo", action="store_true", help="...")
    #   --demo           → 标志名，命令行中用 --demo 传入
    #   action="store_true" → 如果用户提供了 --demo，args.demo 为 True；否则默认 False
    #   help="..."       → 运行 python orchestrator.py --help 时显示的说明文字
    p.add_argument("--demo", action="store_true", help="run a tiny MoE demo")
    args = p.parse_args()

    # ─── 阶段 1：运行单元测试（始终执行）───
    # 三个测试文件分别验证 Part 5 的核心模块：
    #   test_gate_shapes.py  → gating.py：门控输出形状、top-k 选择是否正确
    #   test_moe_forward.py  → moe.py：MoE 层的前向传播 dispatch/combine 逻辑
    #   test_hybrid_block.py → block_hybrid.py：混合 dense+MoE block 的组合行为
    # pytest -q（--quiet）减少输出噪音，只显示关键信息
    run("python -m pytest -q tests/test_gate_shapes.py")
    run("python -m pytest -q tests/test_moe_forward.py")
    run("python -m pytest -q tests/test_hybrid_block.py")

    # ─── 阶段 2：可选 MoE 演示（仅 --demo 时执行）───
    # 演示用一个小型 MoE 模型跑一次前向传播，输出 token 到专家的路由分布直方图，
    # 直观展示"哪些专家被选中的多、哪些少"——这对理解负载均衡非常重要。
    # 参数说明：
    #   --tokens 6    → 模拟 6 个 token（序列长度 = 6）
    #   --hidden 128  → 隐藏维度 = 128
    #   --experts 4   → 4 个专家（MLP）
    #   --top_k 1     → 每个 token 只选 1 个专家（最典型的稀疏 MoE 设置）
    if args.demo:
        run("python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1")

    # 全部通过后的成功提示
    # ✅ emoji 让输出更加友好，一眼就能看到"全绿"
    print("\nPart 5 checks complete. ✅")

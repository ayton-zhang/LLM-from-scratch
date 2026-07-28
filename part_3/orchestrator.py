# ==========================================
# Orchestrator：Part 3 测试运行器 + 生成演示
# ==========================================
# 本文件的职责：
#   1. 一键运行 Part 3 的所有单元测试
#   2. 可选运行一个小型文本生成 demo（验证各组件端到端协同）
#
# 为什么需要这个文件？Part 3 拆成了多个独立模块（RMSNorm、RoPE、SwiGLU、
# KV Cache、注意力、Block、模型），单独运行某个测试不能验证它们之间的协作。
# orchestrator 把测试 + demo 串起来，一次确认所有组件拼在一起没问题。
#
# 用法：
#   cd part_3
#   python orchestrator.py           # 只跑测试
#   python orchestrator.py --demo    # 跑测试 + 生成 demo
#   pytest -q                        # 等价于跑测试（pytest 直接收集 tests/）

# Repository layout (Part 3)
#
#   part_3/
#     orchestrator.py              # runs tests + a small generation demo
#     tokenizer.py                 # local byte-level tokenizer (self-contained)
#     rmsnorm.py                   # 3.1 RMSNorm
#     rope.py                      # 3.2 RoPE cache + apply
#     swiglu.py                    # 3.3 SwiGLU FFN
#     kv_cache.py                  # 3.4/3.6 KV cache + rolling buffer
#     attn_modern.py               # attention w/ RoPE, sliding window, sink, optional KV cache
#     block_modern.py              # block = (RMSNorm|LN) + modern attention + (SwiGLU|GELU)
#     model_modern.py              # GPTModern wrapper with feature flags
#     demo_generate.py             # simple generation demo (shows KV cache + sliding window)
#     tests/
#       test_rmsnorm.py
#       test_rope_apply.py
#       test_kvcache_shapes.py
#
# Run from inside `part_3/`:
#   cd part_3
#   python orchestrator.py --demo
#   pytest -q

# 标准库导入：
#   argparse   : 解析命令行参数（--demo）
#   pathlib    : 面向对象的文件路径操作，比 os.path 更现代
#   subprocess : 在子进程中运行外部命令（python -m pytest）
#   sys        : 获取 Python 解释器路径、控制程序退出
#   shlex      : 按 shell 语法把命令字符串拆成参数列表（安全地处理带空格的参数）
import argparse, pathlib, subprocess, sys, shlex

# ROOT：本文件所在目录的绝对路径，即 part_3/。
# 语法：pathlib.Path(__file__) 是本文件的路径对象。
#       .resolve() 把相对路径转为绝对路径（消除符号链接）。
#       .parent 取父目录。
# 所有子进程的 cwd 都设为 ROOT，确保无论从哪里调用 orchestrator.py，
# 测试和 demo 都能正确找到 part_3/ 下的模块。
ROOT = pathlib.Path(__file__).resolve().parent

# ==========================================
# 工具函数：在子进程中运行命令
# ==========================================
def run(cmd: str):
    """在 ROOT 目录下运行一条 shell 命令，失败时退出整个程序。

    为什么要用 subprocess 而不是直接 import 测试模块来跑？
    - 每个测试文件是独立的进程，互不污染（全局状态、缓存不会被残留）。
    - 行为等价于用户在终端手动敲命令，更真实地模拟实际使用场景。
    - demo_generate.py 是独立脚本，不是模块，必须用子进程运行。
    """
    # 打印将要执行的命令，让用户清楚每步在干什么。
    print(f"\n>>> {cmd}")

    # shlex.split(cmd)：按 shell 语法把整条命令拆成参数列表。
    # 例：'python -m pytest -q tests/test_foo.py' → ['python', '-m', 'pytest', '-q', 'tests/test_foo.py']
    # 比手动 .split(' ') 安全：shlex 能正确处理引号内的空格、转义字符等边界情况。
    # cwd=ROOT：子进程的工作目录设为 part_3/，确保相对路径引用正确。
    #
    # 语法：sys.executable 返回当前 Python 解释器的绝对路径（如
    #   /home/yuteng/LLM-from-scratch/.venv/bin/python）。
    # 为什么不用硬编码的 "python"？子进程的 PATH 环境变量可能与当前终端不同
    # （如 virtualenv 或 IDE 内置终端），PATH 中可能没有 python。
    # sys.executable 始终指向正确解释器，避免 "python: command not found"。
    args = shlex.split(cmd)
    if args and args[0] in ("python", "python3"):
        args[0] = sys.executable
    res = subprocess.run(args, cwd=ROOT)

    # sys.exit(n)：以退出码 n 终止整个 Python 进程。
    # n=0 表示正常退出，n≠0 表示异常退出。这里把子进程的退出码透传出去，
    # 这样外部脚本（如 CI）能正确感知测试是否失败。
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 主流程
# ==========================================
# 语法：`if __name__ == "__main__":` 是 Python 的标准入口守卫。
# 当 `python orchestrator.py` 时，__name__ 被 Python 设为 "__main__"，执行 if 块。
# 当 `import orchestrator` 时，__name__ 是 "orchestrator"，if 块不执行。
# 这样同一个文件既可以当脚本跑，也可以被别的模块 import 而不触发执行。
if __name__ == "__main__":
    # argparse.ArgumentParser() 创建一个命令行参数解析器。
    # 把用户敲的 --demo 之类选项转成结构化的 Python 对象，无需手写 if '--demo' in sys.argv。
    p = argparse.ArgumentParser()
    # --demo 默认启用：直接 debug orchestrator.py 就会走完整流程（测试 + 生成演示），
    # 覆盖所有组件的 forward / backward / KV Cache 细节，适合学习时单步跟踪。
    # 如果只想跑测试、跳过生成演示，加 --skip-demo 即可。
    p.add_argument("--skip-demo", action="store_true",
                   help="skip the generation demo, run tests only")
    # p.parse_args() 解析 sys.argv（命令行参数列表），返回包含解析结果的命名空间对象。
    args = p.parse_args()

    # 1) run unit tests
    # 共三个测试文件，分别验证：
    #   test_rmsnorm.py      → RMSNorm 形状正确、归一化行为合理
    #   test_rope_apply.py   → RoPE 旋转后形状不变、旋转角度的正确性
    #   test_kvcache_shapes.py → KVCache 和 RollingKV 的形状与裁剪行为
    #
    # 如果任何测试失败，run() 内部会调用 sys.exit(非0) 终止程序，
    # 后续的测试和 demo 不会继续执行（fail-fast 策略，快速暴露问题）。
    run(f"{sys.executable} -m pytest -q tests/test_rmsnorm.py")
    run(f"{sys.executable} -m pytest -q tests/test_rope_apply.py")
    run(f"{sys.executable} -m pytest -q tests/test_kvcache_shapes.py")

    # 2) (optional) generation demo
    # --demo 是可选的：不加这个参数只跑测试，加了才跑生成 demo。
    # demo 参数解释：
    #   --rmsnorm --rope --swiglu : 启用三个现代组件（RMSNorm 归一化、
    #                                 RoPE 位置编码、SwiGLU FFN）
    #   --sliding_window 64       : 滑动窗口大小 = 64，每个 token 只看最近 64 个历史
    #   --sink 4                  : attention_sink = 4，保留开头 4 个 token 始终可见
    #   --tokens 200              : 生成 200 个新 token
    # 共计生成 prompt(5) + 200 = 205 个 token，远超窗口 64，可以观察滑动窗口
    # 裁剪效果 + attention sink 保留开头 token 的行为。
    if not args.skip_demo:
        run(f"{sys.executable} demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    # 全部通过，打印成功标记。
    print("\nPart 3 checks complete. ✅")

# ==========================================
# Part 6：监督微调 (SFT — Supervised Fine-Tuning)
# ==========================================
# 本模块实现了一个完整的 SFT 流程：将预训练好的语言模型在"指令→回答"格式的
# 对话数据上继续训练，使其学会"听懂人类指令并按格式作答"的能力。
#
# 与预训练（Part 4）的关键区别：
#   - 预训练：在海量无标签文本上学习"下一个 token 是什么"（通用语言能力）
#   - SFT：在少量高质量"对话对"上微调，教会模型听懂任务指令并给出合理回答
#
# Repository layout (Part 6)
#
#   part_6/
#     orchestrator.py           # run unit tests + optional tiny SFT demo
#     formatters.py             # 6.1 prompt/response templates
#     dataset_sft.py            # HF dataset loader (+tiny fallback) → (prompt, response)
#     collator_sft.py           # 6.2 causal LM labels with masking
#     curriculum.py             # 6.3 length‑based curriculum sampler
#     evaluate.py               # 6.4 simple exact/F1 metrics
#     train_sft.py              # minimal one‑GPU SFT loop (few steps)
#     sample_sft.py             # load ckpt & generate from instructions
#     tests/
#       test_formatter.py
#       test_masking.py
#
# Run from inside `part_6/`:
#   cd part_6
#   python orchestrator.py --demo
#   pytest -q

#
#   part_6/
#     orchestrator.py           # run unit tests + optional tiny SFT demo
#     formatters.py             # 6.1 prompt/response templates
#     dataset_sft.py            # HF dataset loader (+tiny fallback) → (prompt, response)
#     collator_sft.py           # 6.2 causal LM labels with masking
#     curriculum.py             # 6.3 length‑based curriculum sampler
#     evaluate.py               # 6.4 simple exact/F1 metrics
#     train_sft.py              # minimal one‑GPU SFT loop (few steps)
#     sample_sft.py             # load ckpt & generate from instructions
#     tests/
#       test_formatter.py
#       test_masking.py
#
# Run from inside `part_6/`:
#   cd part_6
#   python orchestrator.py --demo
#   pytest -q

### FILE: part_6/orchestrator.py

# ==========================================
# 导入模块：编排 SFT 训练、测试和采样流程
# ==========================================
# argparse    → 解析命令行参数（--demo）
# pathlib     → 面向对象的路径操作，比 os.path 更直观
# subprocess  → 在 Python 中调用子进程（运行 pytest、train_sft.py 等）
# sys         → 退出程序时返回错误码
# shlex       → 按 shell 语法规则拆分命令字符串（正确处理引号、转义等）
import argparse, pathlib, subprocess, sys, shlex

# ─── 项目根目录定位 ───
# 语法：__file__ 是当前脚本的文件路径；.resolve() 转换为绝对路径；.parent 取父目录。
# 最终 ROOT 指向 part_6/ 目录，所有子进程命令都以此为工作目录执行，
# 确保无论从哪个目录调用本脚本，相对路径引用（如 tests/、train_sft.py）都不会出错。
ROOT = pathlib.Path(__file__).resolve().parent

# ==========================================
# run()：统一封装子进程调用
# ==========================================
# 职责：接收一条 shell 命令字符串，在 ROOT 目录下执行，失败时立即退出。
# 这样主逻辑中只需调用 run("python train_sft.py ...") 即可，无需重复写
# subprocess 的样板代码和错误检查逻辑。
def run(cmd: str):
    # 先打印命令，方便调试时看到每一步执行了什么
    print(f"\n>>> {cmd}")

    # shlex.split(cmd) 将字符串按 shell 语法拆分为列表，例如：
    #   "python -m pytest -q tests/test.py" → ["python", "-m", "pytest", "-q", "tests/test.py"]
    # 比手写 cmd.split() 更安全：能正确处理引号包裹的参数（如 --prompt 'What is DNA?'）
    # cwd=ROOT 指定子进程的工作目录为 part_6/，保证相对路径正确
    args = shlex.split(cmd)
    # 将 "python" 替换为 sys.executable（当前 Python 解释器的绝对路径），
    # 避免 bare "python" 在 PATH 中找不到的问题（如 venv、conda 环境）。
    if args and args[0] == "python":
        args[0] = sys.executable
    res = subprocess.run(args, cwd=ROOT)

    # 子进程返回码非 0 表示执行失败（测试不通过、脚本报错等），
    # 此时用 sys.exit() 终止整个编排流程，并将错误码向上传递，
    # 这样 CI/CD 或调用脚本能感知到失败而非静默通过。
    if res.returncode != 0:
        sys.exit(res.returncode)

# ==========================================
# 主入口：编排 Part 6 的完整验证流程
# ==========================================
# 语法：`if __name__ == "__main__"` 是 Python 的惯用法，
# 表示"仅当直接运行本文件时才执行以下代码，被 import 时不执行"。
# 这样其他模块可以安全地 import 本文件中的 run() 函数而不触发测试流程。
if __name__ == "__main__":
    # ─── 命令行参数解析 ───
    # argparse 自动生成 --help 帮助信息，提升脚本的可用性
    p = argparse.ArgumentParser()
    # 默认 demo=True：方便 debug 模式下一键运行完整 SFT 流程，无需手动传 --demo。
    # 如需跳过 demo（只跑单元测试），显式传入 --no-demo 即可。
    p.add_argument("--demo", action="store_true", default=True, help="tiny SFT demo on a few samples")
    p.add_argument("--no-demo", action="store_false", dest="demo", help="skip the SFT demo, run tests only")
    args = p.parse_args()

    # ==========================================
    # 第一步：运行单元测试
    # ==========================================
    # 无论是否开启 demo，单元测试必须通过——这是代码质量的底线保证。
    # pytest -q（quiet 模式）减少输出噪音，只看关键结果。

    # test_formatter.py：验证 6.1 节的 prompt/response 模板是否正确格式化
    run("python -m pytest -q tests/test_formatter.py")
    # test_masking.py：验证 6.2 节的 causal LM label masking 是否正确
    #   （训练时 prompt 部分不计算 loss，只对 response 部分做监督）
    run("python -m pytest -q tests/test_masking.py")

    # ==========================================
    # 第二步（可选）：运行 SFT 完整 demo
    # ==========================================
    # 当用户传入 --demo 标志时执行，展示从训练到推理的完整 SFT 流程。
    # Demo 流程分为两个阶段：
    #   阶段 A：SFT 训练 —— 加载 Part 4 的预训练权重，在对话数据上微调
    #   阶段 B：指令采样 —— 用训练好的模型回答几个示例问题，验证效果
    if args.demo:
        # --ckpt ../part_4/runs/part4-demo/model_last.pt # assumes Part 4 demo has been run

        # ═══════════════════════════════════════════
        # 阶段 A：SFT 训练（train_sft.py）
        # ═══════════════════════════════════════════
        # 各参数含义：
        #   --data huggingface  → 从 HuggingFace 加载 SFT 数据集（带 tiny fallback）
        #   --ckpt ...          → 加载 Part 4 预训练好的基础模型权重作为起点
        #                         （SFT 是在已有语言能力上做"方向微调"，不是从头训练）
        #   --out runs/sft-demo → 训练输出目录（保存 checkpoint 和日志）
        #   --steps 300         → 训练步数；SFT 数据量远小于预训练，几步即可见效
        #   --batch_size 8      → 每步的样本数
        #   --block_size 256    → 最大序列长度（token 数），超长序列会被截断
        #   --n_layer 2         → Transformer 层数（与 Part 4 的预训练模型架构一致）
        #   --n_head 2          → 注意力头数
        #   --n_embd 128        → 嵌入维度
        run("python train_sft.py --data huggingface --ckpt ../part_4/runs/part4-demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128")

        # ═══════════════════════════════════════════
        # 阶段 B：指令采样（sample_sft.py）—— 用训练好的模型回答问题
        # ═══════════════════════════════════════════
        # 对三个不同类型的 prompt 分别生成回答，覆盖：
        #   1. 简单事实问答（三原色）
        #   2. 缩写释义问答（DNA）
        #   3. 代码理解与改写（逆向工程 factorialize 函数）
        # 这些用例覆盖了 SFT 训练集的主要任务类型，能较全面地验证微调效果。

        # ─── 示例 1：简单事实问答 ───
        # temperature=0.2 表示低随机性采样，输出更确定、更"保守"，
        # 适合事实型问题（不希望模型胡说八道，要给出准确答案）
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What are the three primary colors?' --tokens 30 --temperature 0.2")

        # ─── 示例 2：缩写释义问答 ───
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What does DNA stand for?' --tokens 30 --temperature 0.2")

        # ─── 示例 3：代码理解与改写 ───
        # 这是一个更复杂的任务：给模型一段有 bug 的代码（factorialize 的循环少乘了 num），
        # 让它"逆向工程并创建新版本"，考验模型的代码理解和生成能力。
        # --tokens 64 给更长的输出空间（代码生成通常比简短回答需要更多 token）
        # 语法注意：prompt 字符串中包含 \n 转义字符，作为 Python 字面量时会被解释为换行符
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'Reverse engineer this code to create a new version\ndef factorialize(num):\n  factorial = 1\n  for i in range(1, num):\n    factorial *= i\n  \n  return factorial' --tokens 64 --temperature 0.2")

    # ─── 全部完成 ───
    # 如果代码执行到这里，说明：
    #   1. 所有单元测试通过（第一步未触发 sys.exit）
    #   2. 如果开启了 --demo，训练和采样也都成功完成
    print("\nPart 6 checks complete. ✅")

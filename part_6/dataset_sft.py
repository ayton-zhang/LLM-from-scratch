# ==========================================
# Part 6.1：SFT 数据集加载器 — HuggingFace 数据集 + 本地 fallback
# ==========================================
# 本模块负责加载 SFT（监督微调）所需的对话数据集。
# SFT 的数据格式与预训练完全不同：
#   - 预训练：海量原始文本（如维基百科、书籍），无结构，只要求"像人写的"
#   - SFT：(指令, 回答) 对，结构化数据，要求模型学会"听指令→给答案"
#
# 数据来源有两层保障：
#   1. 首选：通过 HuggingFace `datasets` 库加载 `tatsu-lab/alpaca` 数据集
#      （Alpaca 是斯坦福发布的经典指令微调数据集，含 52K 条指令-回答对）
#   2. 兜底：如果 `datasets` 库无法导入或网络不可用，使用硬编码的 3 条简单示例
#      （确保代码在任何环境下都能跑通 demo 流程）

from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import traceback

# ==========================================
# 可选依赖：HuggingFace `datasets` 库
# ==========================================
# `datasets` 是 HuggingFace 生态的核心组件，提供了统一的 API 来加载
# 成千上万个开源数据集。但它是一个"重"依赖（下载量大、需要网络），
# 因此这里用 try/except 做优雅降级：能装就装，装不了也不报错，
# 后续用本地 fallback 数据继续跑。
try:
    # 语法：from ... import ...，将 load_dataset 函数导入当前命名空间。
    # load_dataset 是 HuggingFace datasets 库的核心入口，用法：
    #   ds = load_dataset("数据集名", split="train[:200]")
    # 它自动处理下载、缓存、切片、流式读取等所有细节。
    from datasets import load_dataset
except Exception:
    # 导入失败时不崩溃，只是打印提示并将 load_dataset 设为 None。
    # 后续代码通过 `if load_dataset is not None` 判断是否可用，
    # 这是一种"功能降级（graceful degradation）"的设计模式。
    print("Couldn't import `datasets`. Will use fallback data only.")
    load_dataset = None

from formatters import Example

# ==========================================
# SFTItem：一条 SFT 训练样本的数据结构
# ==========================================
# 语法：@dataclass 是 Python 3.7+ 的数据类装饰器。
# 它自动生成 __init__、__repr__、__eq__ 等样板方法，让类定义更简洁。
# 等价于手写：
#   class SFTItem:
#       def __init__(self, prompt: str, response: str):
#           self.prompt = prompt
#           self.response = response
# 但 @dataclass 还额外提供了可读性更好的 __repr__ 和值相等的 __eq__。
@dataclass
class SFTItem:
    """一条 SFT 训练样本：prompt 是用户指令，response 是期望的模型回答。"""
    prompt: str     # 用户的指令/问题，例如 "What are the three primary colors?"
    response: str   # 期望模型生成的回答，例如 "Red, yellow, and blue."


# ==========================================
# load_tiny_hf()：加载 SFT 数据集（HF 优先 + 本地 fallback）
# ==========================================
# 参数说明：
#   split : HuggingFace 数据集切片字符串，默认 "train[:200]" 表示训练集的前 200 条。
#           语法来自 HuggingFace datasets 库，支持：
#             "train[:200]"   → 前 200 条
#             "train[50:100]" → 第 50 到 99 条
#             "train[-10:]"   → 最后 10 条
#   sample_dataset : 布尔标志，True 时跳过 HF 加载、直接使用 fallback 数据。
#                    用于测试场景——不依赖网络和 datasets 库即可验证 collator 逻辑。
#
# 返回值：
#   List[SFTItem]：SFT 样本列表，每个元素包含 .prompt 和 .response。
#
# 设计思路（fallback 模式）：
#   深度学习项目经常依赖外部数据和库，但 demo/测试环境可能没有网络或 GPU。
#   "HF 优先 + 本地兜底"的策略让代码在多种环境下都能跑，减少"it works on my machine"的尴尬。
def load_tiny_hf(split: str = "train[:200]", sample_dataset: bool = False) -> List[SFTItem]:
    """Try to load a tiny instruction dataset from HF; fall back to a baked-in list.
    We use `tatsu-lab/alpaca` as a familiar schema (instruction, input, output) and keep only a slice.
    """

    # ─── 第一步：尝试从 HuggingFace 加载 ───
    # load_dataset is not None → datasets 库成功导入
    # not sample_dataset     → 用户没有要求跳过 HF（测试模式下 sample_dataset=True）
    items: List[SFTItem] = []
    if load_dataset is not None and not sample_dataset:
        try:
            # load_dataset("tatsu-lab/alpaca", split=split) 从 HF Hub 拉取 Alpaca 数据集。
            # Alpaca 数据集的结构：每条包含 instruction（指令）、input（可选输入）、output（回答）。
            # split 参数控制取多少数据——这里默认只取前 200 条，足够 demo 使用。
            ds = load_dataset("tatsu-lab/alpaca", split=split)

            # 遍历数据集中的每一行，将 Alpaca 格式转换为 SFTItem 格式。
            # ds 是一个可迭代对象（类似列表），每行是一个 Python 字典。
            for row in ds:
                # Alpaca 数据有三列：instruction（任务描述）、input（附加上下文）、output（答案）。
                # .get(key, default) 是字典的安全取值方法：key 存在返回值，不存在返回 default。
                # 这里 default="" 避免了 KeyError（某些数据集的字段名可能略有不同）。
                # .strip() 去掉字符串首尾的空白字符（空格、换行等），保持数据干净。
                instr = row.get("instruction", "").strip()  # 例如 "Translate to French"
                inp = row.get("input", "").strip()          # 例如 "Hello, how are you?"
                out = row.get("output", "").strip()         # 例如 "Bonjour, comment allez-vous?"

                # 如果存在 input（附加上下文），将其拼接到 instruction 后面。
                # 这样模型看到的是一个完整的问题："Translate to French\nHello, how are you?"
                # 而不是分开的两段信息。\n 作为自然分隔符，模型在预训练中已经学会了理解。
                if inp:
                    instr = instr + "\n" + inp

                # 过滤掉空指令或空回答的无效样本（数据集中可能偶有残缺条目）。
                # 布尔上下文中的空字符串为 False：
                #   instr=""  → False  → 跳过
                #   instr="x" → True   → 保留
                if instr and out:
                    items.append(SFTItem(prompt=instr, response=out))

        except Exception:
            # ─── HF 加载失败时的处理 ───
            # 可能的失败原因：
            #   1. 网络不可用（无法连接 HuggingFace Hub）
            #   2. 磁盘空间不足（无法缓存数据集）
            #   3. 数据集 schema 变更（字段名不匹配）
            # 任何异常都静默处理（pass），不打印堆栈——因为下面有 fallback 兜底，
            # 这不是致命错误，不需要吓到用户。
            pass

    # ─── 第二步：如果上面一条数据都没拿到，用硬编码的 fallback ───
    # 注意：HF 加载可能返回空列表（如 split 范围超出数据集），
    # 也可能因异常被跳过，两种情况下 items 都是空的，走 fallback。
    if not items:
        # fallback tiny list
        # 3 条极简示例，覆盖不同类型的问答：
        #   1. 事实问答   → "First prime number" → "2"
        #   2. 常识问答   → "What are the three primary colors?" → "red"
        #   3. 物品识别   → "Device name which points to direction?" → "compass"
        # 注意：fallback 的回答故意不完整（三原色只答了 red），
        # 这是为了展示 SFT 的局限性——数据质量决定模型质量。
        seeds = [
            ("First prime number", "2"),
            ("What are the three primary colors?", "red"),
            ("Device name which points to direction?", "compass"),
        ]
        # 语法：列表推导式 [SFTItem(prompt=p, response=r) for p, r in seeds]
        # 等价于：
        #   items = []
        #   for p, r in seeds:
        #       items.append(SFTItem(prompt=p, response=r))
        # 元组解包 for p, r in seeds：seeds 的每个元素是 (str, str) 元组，
        # 循环时自动解包为 p 和 r 两个变量。
        items = [SFTItem(prompt=p, response=r) for p, r in seeds]

    return items

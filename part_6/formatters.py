# ==========================================
# Part 6.1：对话格式化器 — Prompt/Response 模板
# ==========================================
# 本模块定义了 SFT 训练和推理中最基础的一层：**对话模板**。
#
# 为什么需要模板？
#   原始数据是裸文本对，如 ("What is DNA?", "Deoxyribonucleic acid")。
#   如果直接拼接喂给模型，模型不知道哪段是"指令"、哪段是"回答"，
#   也不知道何时该自己生成、何时该等待用户输入。
#
#   模板的作用就是在文本中加入**角色标记**（类似剧本中的 "User:" "Assistant:"），
#   让模型学会识别：
#     - "### Instruction: ..."  → 这是用户指令，我需要理解它
#     - "### Response: "        → 轮到我了，我要开始生成回答了
#
# 模板设计原则：
#   1. 清晰的分隔标记：让模型在 token 级别就能区分"指令"和"回答"区域
#   2. 训练和推理一致：format_prompt_only 在推理时生成与训练时完全相同的前缀格式，
#      确保模型看到的上下文分布不变
#   3. 特殊 token（<s>）：句子开头标记，帮助模型识别序列边界

"""Prompt/response formatting utilities (6.1).
We keep a very simple template with clear separators.
"""
from dataclasses import dataclass


# ==========================================
# 对话模板：定义 SFT 的文本格式
# ==========================================
# 这个模板借鉴了 Stanford Alpaca 数据集的格式，设计思路：
#   - <s> 开头标记：表示一个完整训练样本的开始（"start of sequence"），
#     帮助模型区分不同训练样本的边界。在 BPE/Llama 等 tokenizer 中，
#     <s> 通常是特殊 token，有独立的 token ID。
#   - ### Instruction: 作为指令区域的标题，模型看到它就知道下面是用户指令
#   - {instruction} 占位符：Python 字符串格式化的槽位，运行时填入实际指令文本
#   - ### Response: 作为回答区域的标题，告诉模型"从这里开始是你的回答"
#   - {response} 占位符：填入期望的模型回答
#   - </s> 结尾标记：表示样本结束（"end of sequence"），告诉模型"回答到此为止"
#
# 语法：Python 的多行字符串用三引号 """...""" 或 '''...''' 包裹。
# 这里用括号包裹多行字符串（隐式拼接），每一行末尾的 \n 显式换行。
#
# 格式化后的完整文本示例：
#   <s>
#   ### Instruction:
#   What are the three primary colors?
#
#   ### Response:
#   Red, yellow, and blue.</s>
#
template = (
    "<s>\n"                                   # 句子开头标记（BOS token）
    "### Instruction:\n{instruction}\n\n"      # 指令区域：### Instruction: 标题 + 内容
    "### Response:\n{response}</s>"            # 回答区域：### Response: 标题 + 内容 + 结尾标记
)

# ==========================================
# Example：一条格式化后的 SFT 样本
# ==========================================
# 语法：@dataclass 装饰器自动生成 __init__、__repr__、__eq__ 等方法。
# 与 dataset_sft.py 中的 SFTItem 作用类似：都是 (指令, 回答) 对的数据容器。
# 区别：
#   - SFTItem 存原始文本（用于数据加载阶段）
#   - Example 存处理后的文本（用于格式化阶段，已经过 .strip() 清理）
#   两者分离使得数据处理管线更清晰——加载 → 清理 → 格式化，各阶段职责分明。
@dataclass
class Example:
    """一条格式化后的 SFT 样本：instruction 是用户指令，response 是期望回答。"""
    instruction: str   # 用户指令，例如 "What is DNA?"
    response: str      # 模型回答，例如 "Deoxyribonucleic acid"


# ==========================================
# format_example()：生成完整的 SFT 训练文本
# ==========================================
# 这是训练时使用的格式化函数，将一条 (instruction, response) 对
# 组装成完整的对话文本，作为模型的输入序列。
#
# 完整文本包含 instruction 和 response 两部分，模型从头看到尾，
# 但在 collator_sft.py 中只对 response 部分计算 loss（prompt masking）。
#
# 参数：
#   ex: Example 对象，包含 instruction 和 response 字段
#
# 返回：
#   str：完整的格式化对话文本，可直接喂给 tokenizer
def format_example(ex: Example) -> str:
    # 语法：str.format(**kwargs) 是 Python 字符串格式化方法。
    # template 中定义了 {instruction} 和 {response} 两个占位符，
    # format(instruction=..., response=...) 将实际值填入占位符。
    #
    # .strip() 去除首尾空白字符（空格、换行、制表符等），
    # 防止数据中的多余空白破坏模板格式。例如原始 response 是
    # "  Red, yellow, and blue.  " → 清理后为 "Red, yellow, and blue."
    return template.format(
        instruction=ex.instruction.strip(),
        response=ex.response.strip()
    )


# ==========================================
# format_prompt_only()：生成仅含指令的前缀文本（推理用）
# ==========================================
# 这是推理时使用的格式化函数，与 format_example 的关键区别：
#   response 参数设为空字符串 ""。
#
# 效果：生成的文本以 "### Response:\n" 结尾，**后面没有任何内容**。
# 模型看到这个半截文本后，会"意识到"现在该自己接续生成了，
# 于是开始自回归地输出回答内容。
#
# 类比：format_prompt_only 像一张"请作答"的试卷——题目（instruction）已经印好，
# 答题区（Response: 后面）是空白的，等模型来填写。
#
# 为什么必须保持与训练时相同的格式？
#   模型在 SFT 训练中反复见到 "### Response:\n{实际回答}</s>" 的模式，
#   它学会了"看到 ### Response:\n 就开始生成内容"的行为习惯。
#   如果推理时换成其他格式（如 "Q: ... A: "），模型会"懵"——它在训练中从未见过这种模式。
#
# 参数：
#   instruction: str，用户输入的指令文本（原始，未格式化）
#
# 返回：
#   str：以 "### Response:\n" 结尾的格式化前缀文本，模型将在此之后接续生成
def format_prompt_only(instruction: str) -> str:
    # response="" → 模板中的 {response} 被替换为空字符串，
    # 因此输出以 "</s>" 结尾前没有任何回答内容。
    # 实际效果："### Response:\n</s>" —— 注意 </s> 紧跟在换行后。
    # collator_sft.py 中会通过 .replace('</s>','') 去掉结尾标记，
    # 让模型从纯 "### Response:\n" 后开始生成。
    return template.format(instruction=instruction.strip(), response="")

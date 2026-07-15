# ==========================================
# Part 6.3：课程学习采样器 — 由短到长的训练策略
# ==========================================
# 课程学习（Curriculum Learning）是 Bengio 等人 2009 年提出的训练策略，
# 核心思想：模仿人类学习过程——先学简单的，再逐步增加难度。
# 在 NLP 中，"难度"通常用序列长度来衡量：短文本比长文本更容易学习。
#
# 为什么 SFT 阶段特别适合课程学习？
#   - SFT 数据长度差异巨大：有的问答只有几个词，有的涉及长代码。
#     如果随机打乱，模型刚被长样本"折磨"完（loss 巨大），转头又去处理短样本，
#     训练曲线会剧烈震荡，收敛变慢。
#   - 从短到长的排序让模型先建立"听懂指令→给出简短回答"的基本能力，
#     然后逐步扩展到更长、更复杂的回复，训练更平稳。
#
# 与随机采样（shuffle）的对比：
#   - 随机采样：每步从数据集中随机取 batch，简单直接，但长样本可能让早期 loss 爆炸
#   - 课程学习：按长度排序后顺序取，训练更平滑，但可能引入顺序偏差（模型过拟合
#     最后几条长样本）。本实现只用一轮（one pass）来减轻此问题。

from __future__ import annotations
from typing import List


# ==========================================
# LengthCurriculum：按 prompt 长度排序的课程采样器
# ==========================================
# 这是一个实现了 Python 迭代器协议（Iterator Protocol）的类，
# 可以被 for 循环直接使用，也可以被 list() 收集为列表。
#
# Python 迭代器协议要求三个方法：
#   __iter__()  → 返回迭代器对象自身（通常是 self），初始化迭代状态
#   __next__()  → 每次调用返回下一个元素；元素耗尽时抛出 StopIteration 异常
#   StopIteration 异常 → 这不是方法，而是 __next__() 必须抛出的信号，
#                          告诉 for 循环"没有更多元素了，可以停止了"
#
# 为什么不用简单的 for item in sorted(items, key=...):
#   封装成类可以在未来扩展功能（如多轮重复、动态重排序、按 batch 分组等），
#   而且让调用方代码更清晰：cur = LengthCurriculum(tuples) 一眼看出意图。
class LengthCurriculum:
    """6.3 Curriculum: iterate examples from short→long prompts (one pass demo)."""

    def __init__(self, items: List[tuple[str, str]]):
        # ─── 核心排序逻辑：按 prompt 长度升序排列 ───
        # items 中的每个元素是 (prompt, response) 元组对。
        # sorted() 是 Python 内置排序函数，返回一个新列表（不修改原列表）。
        #
        # key=lambda p: len(p[0]) 的工作机制：
        #   lambda p: len(p[0])     → 匿名函数，接受一个元组 p，返回 p[0]（prompt 字符串）的长度
        #   sorted(..., key=...)    → 对每个元素调用 key 函数得到"排序键"，按键值升序排列
        #
        # 为什么只按 prompt 长度排序，不按 prompt+response 总长度？
        #   - prompt 是模型"要理解"的部分，较长的 prompt 包含更多上下文和约束条件，
        #     需要模型具备更强的指令理解能力 — 这确实是"更难"的
        #   - response 是模型"要生成"的部分，它的长度影响不大（训练时每个 token 同等对待）
        #   - 两个维度一起排序反而可能把"短 prompt + 长 response"排在前面，
        #     这违背课程学习的初衷（应该先学简单指令）
        #
        # 类比：教小孩做数学题，先教"1+1=?"再教"小明有3个苹果..."的故事题。
        # 题目的"题干长度"决定了理解难度，答案长度不是考虑因素。
        self.items = sorted(items, key=lambda p: len(p[0]))

        # _i 是内部索引指针，指向下一次 __next__() 调用应返回的元素位置。
        # 初始化为 0（从头开始），每次 __next__() 调用后自增 1。
        # 前导下划线 `_i` 是 Python 惯例，表示这是"内部使用"的私有属性，
        # 外部代码不应直接访问它。
        self._i = 0

    # ==========================================
    # __iter__()：初始化迭代状态，返回迭代器自身
    # ==========================================
    # 语法：当 Python 遇到 `for x in obj:` 或 `list(obj)` 时，
    # 首先调用 obj.__iter__() 获取迭代器对象。
    #
    # 这里将 _i 重置为 0，确保每次开始新的迭代都从头遍历。
    # 如果没有这行，连续两次 for x in cur: ... 时第二次会直接空转
    # （因为 _i 已经在上一次循环中被推到末尾了）。
    def __iter__(self):
        self._i = 0         # 重置指针到开头，支持多次迭代
        return self          # 返回自身作为迭代器

    # ==========================================
    # __next__()：返回下一个元素
    # ==========================================
    # for 循环的每一步都会调用此方法获取下一个值。
    #
    # 执行流程：
    #   1. 检查指针是否已越界（_i >= len(items)）
    #   2. 如果越界，抛出 StopIteration 异常 → for 循环感知到并正常退出
    #   3. 如果未越界，取出当前元素，指针前移，返回元素
    #
    # 为什么用 StopIteration 异常而不是 return None？
    #   这是 Python 迭代器协议的硬性规定——StopIteration 是唯一的"迭代结束"信号。
    #   如果 return None，for 循环会把 None 当作一个有效元素（迭代不会停止）。
    #   类比：StopIteration 像红绿灯的"红灯"，告诉遍历循环"停下，结束了"。
    def __next__(self):
        # 语法：len(self.items) 返回列表元素个数；_i 是指针位置。
        # 当指针到达或超过列表长度时，已无更多元素可返回。
        if self._i >= len(self.items):
            # 语法：raise StopIteration 抛出内置异常，告知调用方（for 循环）迭代结束。
            # 虽然在 Python 3.7+ 中 __next__ 也可以 return 来结束，
            # 但显式 raise StopIteration 是标准做法，语义最清晰。
            raise StopIteration

        # 取出当前指针位置的 (prompt, response) 元组
        it = self.items[self._i]
        # 指针前进一位，为下一次 __next__() 调用做准备
        self._i += 1
        return it
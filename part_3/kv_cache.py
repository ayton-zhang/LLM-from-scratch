# ==========================================
# 组件：KV Cache —— 键值缓存（推理加速的核心机制）
# ==========================================
# 背景：标准 Transformer 推理时，每生成一个新 token 都要重新计算所有历史 token 的 K/V，
#       计算量随序列长度 T 平方增长（O(T²)）。生成第 1000 个 token 时要算 1000×1000 次点积，
#       而其中 999 个 token 的 K/V 在上一步就已经算过了——这是巨大的浪费。
#
# KV Cache 的核心思想：
#   把已计算过的 K/V 存起来，下一步只算新 token 的 Q/K/V，
#   Q 与缓存的全部历史 K 做点积得到注意力分数，再对全部历史 V 加权求和。
#   计算量从 O(T²) 降至 O(T)，长序列推理速度大幅提升（10x ~ 100x）。
#
# 类比：考试时先读一遍题目（prefill），之后每题只需回想之前的理解（查缓存），
#       不用每道题都把整张试卷重读一遍。
#
# 本文件提供两种缓存实现：
#   KVCache   : 简单数据容器（@dataclass），只存不裁，适合短序列推理。
#   RollingKV : 滚动缓冲区（sink + window），固定大小裁剪，适合超长序列推理。
from __future__ import annotations
import torch
from dataclasses import dataclass

# ==========================================
# KVCache：简单键值缓存容器
# ==========================================
# KVCache 的角色就是"一个轻量级信封"——只负责装 K 和 V 两个张量，
# 不做任何裁剪、变换或计算。每次 forward 创建新的 KVCache 对象
# （追加了当前 token 的 K/V 后的完整历史），旧对象被丢弃。
#
# 为什么用 @dataclass 而不是普通 class？
#   语法：`@dataclass` 是 Python 装饰器，自动生成 __init__、__repr__、__eq__ 等方法，
#   无需手写 `def __init__(self, k, v): self.k = k; self.v = v`。
#   对于这种"纯数据容器"场景（没有方法逻辑），dataclass 是最简洁的选择。
@dataclass
class KVCache:
    # k 形状 (B, Hk, T, D)：所有历史 token 的 Key。
    #   B=批大小，Hk=KV 头数（GQA 下可能小于 Q 头数），
    #   T=历史 token 总数，D=每头维度 (d_head)。
    k: torch.Tensor  # (B,H,T,D)
    # v 形状 (B, Hk, T, D)：所有历史 token 的 Value，结构与 k 完全对称。
    v: torch.Tensor  # (B,H,T,D)

    # @property 装饰器：让方法像属性一样调用（cache.T 而非 cache.T()），
    # 语义上"缓存的长度"更像属性而非需要计算的函数。
    @property
    def T(self):
        # 返回当前缓存的 token 数（时间维大小），供 model_modern.py 计算 start_pos。
        # start_pos = 0 if kvs[0] is None else kvs[0].T，
        # 告诉 RoPE"新 token 是完整序列的第几个 token"。
        #
        # 语法：.size(2) 取第 2 维（时间维）的大小，等价于 .shape[2]。
        return self.k.size(2)


# ==========================================
# RollingKV：滚动键值缓冲区（StreamingLLM 的缓存策略）
# ==========================================
# 用于超长序列推理（如流式对话、无限长度生成），保证显存固定不增长。
#
# 核心策略（sink + window）：
#   缓冲区只保留两部分——
#     1. 前 sink 个 token（"注意力水槽"，永不丢弃）
#     2. 最近 window 个 token（"滑动窗口"，最新上下文）
#   超出部分（中间的旧 token）被永久丢弃，显存恒定 = O(sink + window)。
#
# 为什么不能只保留 window，必须加 sink？
#   StreamingLLM 论文的发现：LLM 会把大量注意力分数"倾倒"到开头几个 token 上。
#   如果开纯滑动窗口（sink=0），一旦开头 token 被滑出窗口，注意力分布会崩塌，
#   perplexity 急剧飙升。保留开头 sink 个 token 作为"注意力垃圾桶"后，
#   perplexity 在无限长度上保持稳定。
#
# 直觉类比：一个固定容量的笔记本——
#   第一页（sink 锚点）永远不撕，后面的页写满了就覆盖最旧的那页。
class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        # window：滑动窗口大小，最多保留最近 window 个 token 的 K/V。
        # sink：注意力水槽大小，强制保留最开头 sink 个 token 的 K/V，永不丢弃。
        #       sink=0 时退化为纯滑动窗口（只保留最近 window 个），
        #       但正如上面解释的，这会导致推理质量下降。
        self.window = window
        self.sink = sink
        # 初始状态：缓冲区为空（None），第一次 step 时懒初始化。
        # 这样可以避免 __init__ 时就需要知道 K/V 的形状和设备。
        self.k = None
        self.v = None

    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """追加新 token 的 K/V，裁剪到 sink + window 以内，返回裁剪后的值。

        k_new/v_new 形状：(B, Hk, T_new, D)，T_new 通常为 1（decode 阶段）。

        调用方（attn_modern.py）：
          k_all, v_all = kv_cache.step(k, v)  # 追加 + 裁剪，一步完成
        """
        # ─── 第一步：追加新 K/V 到缓冲区末尾 ───
        if self.k is None:
            # 缓冲区为空（第一次调用，通常是 prefill 阶段首次建立缓存），
            # 直接用新 K/V 初始化，不需要拼接。
            self.k, self.v = k_new, v_new
        else:
            # 缓冲区已有数据，把新 token 的 K/V 追加到时间维（dim=2）末尾。
            # 语法：torch.cat([old, new], dim=2) 沿时间维拼接。
            # self.k 形状 (B, Hk, T_old, D) + k_new (B, Hk, T_new, D)
            #   → (B, Hk, T_old + T_new, D)
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        # ─── 第二步：裁剪到 sink + window 以内 ───
        # crop
        if self.k.size(2) > self.window + self.sink:
            # sink_part：开头 sink 个"锚点" token，永不丢弃。
            # 语法：[:, :, :self.sink, :] 取时间维前 sink 个，其余维度全保留。
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]

            # tail_k/tail_v：末尾最近 window 个 token。
            # 语法：[:, :, -self.window:, :] 负索引，取时间维最后 window 个。
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]

            # 重新拼接：[锚点 | 最近窗口]，中间过旧的部分被自然丢弃。
            # 丢弃的是 k[:, :, sink : -window, :] 这一段——既不是锚点，
            # 也不是最近上下文，模型不再需要它们的信息。
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)

        # 返回裁剪后的完整 K/V，供注意力计算使用。
        # 注意：这里返回的是"整个缓冲区的值"，而不仅仅是"刚追加的 k_new"。
        # 调用方用这个返回值直接做 SDPA，无需再手动拼接。
        return self.k, self.v

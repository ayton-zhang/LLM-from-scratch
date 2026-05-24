# ==========================================
# 组件：KVCache / RollingKV —— 键值缓存
# ==========================================
# 背景：标准 Transformer 推理时，每生成一个新 token 都要重新计算所有历史 token 的 K/V，
#       计算量随序列长度 T 平方增长（O(T²)）。
# KV Cache 的核心思想：
#   把已计算过的 K/V 存起来，下一步只算新 token 的 K/V，再拼接到缓存末尾，
#   计算量从 O(T²) 降至 O(T)，长序列推理速度大幅提升。
#
# 本文件提供两种缓存实现：
#   KVCache   : 简单的不可变数据容器（dataclass），适合短序列推理。
#   RollingKV : 滚动缓冲区，配合 sliding_window + attention_sink，适合超长序列推理。
from __future__ import annotations
import torch
from dataclasses import dataclass

# ==========================================
# 组件：KVCache —— 简单键值缓存容器
# ==========================================
# 语法：@dataclass 是 Python 装饰器，自动生成 __init__、__repr__ 等方法，
#       无需手写 def __init__(self, k, v): self.k = k; self.v = v。
# KVCache 是一个"轻量级结构体"，只负责持有 K/V 张量，不做任何裁剪逻辑。
# 每次 forward 返回新的 KVCache 对象（追加了当前 token 的 K/V）。
@dataclass
class KVCache:
    k: torch.Tensor  # (B, H, T, D)：所有历史 token 的 Key，H 为 KV 头数
    v: torch.Tensor  # (B, H, T, D)：所有历史 token 的 Value

    @property
    def T(self):
        # 返回当前缓存的 token 数（序列长度维），用于计算 start_pos。
        # 语法：.size(2) 取第 2 维（时间维）的大小，等价于 .shape[2]。
        return self.k.size(2)

# ==========================================
# 组件：RollingKV —— 滚动键值缓冲区
# ==========================================
# 用于超长序列推理（如流式对话），保证显存占用固定不增长：
#   缓冲区只保留 sink 个"锚点" token + 最近 window 个 token，
#   超出部分自动丢弃（中间的"旧 token"被裁掉）。
# 直觉：就像一个固定容量的笔记本，
#   第一页（attention_sink）永远不擦，后面的页写满了就从最旧的那页开始覆盖。
class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        # window：滑动窗口大小，最多保留最近 window 个 token 的 K/V。
        self.window = window
        # sink：注意力水槽大小，强制保留最开头 sink 个 token 的 K/V，永不丢弃。
        # sink=0 时退化为纯滑动窗口（只保留最近 window 个）。
        self.sink = sink
        # 初始状态缓冲区为空，第一次 step 时懒初始化。
        self.k = None
        self.v = None

    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        # 第一步：把新 token 的 K/V 追加到缓冲区末尾。
        if self.k is None:
            # 缓冲区为空（第一次调用），直接用新 K/V 初始化。
            self.k, self.v = k_new, v_new
        else:
            # 语法：torch.cat([self.k, k_new], dim=2) 在时间维（dim=2）拼接，
            #       把新 token 的 K 追加到历史缓存末尾。
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        # 第二步：如果总长度超出 (sink + window)，裁剪中间过旧的部分。
        # self.k.size(2) 是当前缓存的 token 总数。
        if self.k.size(2) > self.window + self.sink:
            # 保留开头 sink 个"锚点" token（永不丢弃的注意力水槽）。
            # 语法：[:, :, :s, :] 取时间维的前 s 个，其余维度全保留。
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]

            # 保留末尾最近 window 个 token（滑动窗口）。
            # 语法：[:, :, -self.window:, :] 负索引取最后 window 个。
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]

            # 把两段重新拼接：[锚点 | 最近窗口]，中间过旧的部分被自然丢弃。
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val,  tail_v], dim=2)

        # 返回裁剪后的完整 K/V，供注意力计算使用。
        return self.k, self.v
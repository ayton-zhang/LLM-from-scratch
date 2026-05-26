# ==========================================
# 组件：CausalSelfAttentionModern —— 现代因果自注意力
# ==========================================
# 与 Part 2 的 CausalSelfAttention 相比，本模块做了四处关键升级：
#   1. RoPE（旋转位置编码）替换学习型 pos_emb：位置信息编码进 Q/K 旋转角度，外推性更好
#   2. KV Cache：推理时缓存历史 K/V，每步只算新 token，计算量从 O(T²) 降至 O(T)
#   3. 滑动窗口注意力（Sliding Window）：每个 token 只看最近 W 个位置，显存更省
#   4. GQA（分组查询注意力）：多个 Query 头共享同一对 K/V 头，速度更快、显存更少
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        # n_head：Query 的头数，决定模型从多少个"视角"关注序列。
        self.n_head = n_head
        # n_kv_head：K/V 的头数，用于 GQA（分组查询注意力）。
        # 语法：`n_kv_head or n_head` 等价于 `n_kv_head if n_kv_head else n_head`，
        #       当 n_kv_head=None（未指定）时退回到标准 MHA（K/V 头数 = Q 头数）。
        # GQA 直觉：把 8 个 Q 头分成 2 组，每组 4 个 Q 头共享同 1 对 K/V，
        #           K/V 显存减少 4 倍，推理速度更快（LLaMA-3 / Mistral 采用）。
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        # group_size：每组有多少个 Q 头共享同一对 K/V 头。
        # 标准 MHA：group_size=1；MQA（多查询注意力）：group_size=n_head。
        self.group_size = self.n_head // self.n_kv_head
        # d_head：每个注意力头的特征维度，n_embd 均分给所有 Q 头。
        self.d_head = n_embd // n_head

        # GQA 导致 Q 与 K/V 的投影矩阵大小不同，因此分开定义三个线性层。
        # wq 输出 n_head * d_head 维（所有 Q 头），
        # wk/wv 只输出 n_kv_head * d_head 维（更少的 K/V 头），节省参数和显存。
        # bias=False：现代大模型普遍去掉线性层偏置，减少参数量且效果相当。
        self.wq  = nn.Linear(n_embd, self.n_head    * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        # proj：输出投影，把多头拼接后的结果映射回 n_embd 维。
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # use_rope：是否启用旋转位置编码。
        self.use_rope = rope
        # rope_cache 延迟初始化（第一次 forward 时才建立），
        # 因为此时才能确定 device（CPU 还是 GPU）。
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        # sliding_window：局部注意力窗口大小，None 表示全局注意力。
        # 设为整数 W 时每个 token 只看最近 W 个历史 token，
        # 长序列时显存从 O(T²) 降至 O(T·W)（Mistral / Phi 的做法）。
        self.sliding_window = sliding_window
        # attention_sink："注意力水槽"，强制保留序列开头 K 个 token 始终在注意力视野中。
        # 即使开了 sliding_window，这些"锚点"也不会被滑出窗口，
        # 防止模型在极长序列里完全"忘记"开头的重要信息（StreamingLLM 技术）。
        self.attention_sink = attention_sink

    def _maybe_init_rope(self, device):
        # 延迟初始化 RoPECache：只在第一次 forward 时创建，确保 cos/sin 表与模型在同一 device 上。
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        # 语法：B, T, C = x.shape 是元组解包，同时获取批大小、序列长度、隐藏维度。
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # ─── 投影：生成 Q、K、V ───
        # wq(x) 形状 (B, T, n_head*d_head)，reshape 后变 (B, T, n_head, d_head)，
        # .transpose(1, 2) 把头维度提前：(B, n_head, T, d_head)，方便后续批量矩阵乘法。
        q = self.wq(x).view(B, T, self.n_head,    self.d_head).transpose(1, 2)  # (B, H,  T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)  # (B, Hk, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)  # (B, Hk, T, D)

        # ─── RoPE：对当前 token 的 Q/K 施加旋转位置编码 ───
        # 注意：只对"当前这批新 token"做旋转，缓存里的历史 K 在被存入时已经旋转过了，不能重复旋转。
        # start_pos 指定当前 token 在完整序列中的起始位置，确保旋转角度与绝对位置对应。
        if self.use_rope:
            # torch.arange(start_pos, start_pos + T) 生成当前 token 的位置下标序列。
            # 例：start_pos=2, T=1 → pos=[2]，表示当前 token 是序列中第 3 个位置。
            # 例：start_pos=0, T=5 → pos=[0,1,2,3,4]，表示整段 prompt 的从第1个到第5个位置。
            pos = torch.arange(start_pos, start_pos + T, device=x.device)

            # rope_cache.get(pos) 从预计算表里按位置取出对应的 cos/sin 值：
            #   cos/sin 形状：(T, d_head/2)，每行对应一个 token 位置的所有频率维度。
            cos, sin = self.rope_cache.get(pos)

            # apply_rope_single 对 Q/K 做实际的旋转操作：
            #   把每个头的特征向量两两配对，按对应位置的角度旋转，
            #   旋转后 Q·K 的点积自然包含相对位置信息。
            # 注意 V 不做旋转——V 只存语义内容，位置信息只需编码进"谁关注谁"的打分里。
            q = apply_rope_single(q, cos, sin)  # (B, H,  T, D)
            k = apply_rope_single(k, cos, sin)  # (B, Hk, T, D)

        # ─── KV Cache：拼接历史缓存 ───
        # 推理时 kv_cache 已存有之前所有 token 的 K/V，
        # 语法：torch.cat([a, b], dim=2) 在时间维（dim=2）拼接，追加新 token 的 K/V。
        # k_all 形状：(B, Hk, Tpast+T, D)，包含完整历史。
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B, Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # ─── 滑动窗口 + 注意力水槽裁剪 ───
        # 当历史长度超过 (sliding_window + attention_sink) 时，裁剪 K/V，只保留：
        #   前 attention_sink 个 token（锚点，永不丢弃）
        # + 最近 sliding_window 个 token（滑动窗口）
        # 语法：torch.cat([..., ...], dim=2) 把两段重新拼成完整的局部上下文。
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # ─── GQA 扩展：把 Hk 个 K/V 头复制成 H 个，与 Q 头数对齐 ───
        # 语法：.repeat_interleave(group_size, dim=1) 沿头维度（dim=1）将每个 K/V 头
        #       重复 group_size 次（交错复制，而非整体拼接），
        #       例如 [k0, k1] 扩展为 [k0, k0, k0, k0, k1, k1, k1, k1]（group_size=4）。
        # 标准 MHA（n_kv_head == n_head）跳过此步，直接使用原始 K/V。
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, D)
        else:
            k_attn, v_attn = k_all, v_all

        # ─── 缩放点积注意力 ───
        # F.scaled_dot_product_attention 是 PyTorch 2.0+ 的融合算子，
        # 内部自动完成 scale（除以 √d_head）、softmax、dropout、与 V 的加权求和，
        # 比手写拆开算更快（支持 Flash Attention 等底层优化）。
        # is_causal=True：训练时开启因果掩码（下三角注意力），防止未来 token 信息泄露；
        # is_causal=False：推理时 kv_cache 已存在，当前只有 1 个新 token，无需因果掩码。
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)  # (B, H, T, D)

        # 把多头结果拼回单向量：
        # .transpose(1, 2)：(B, H, T, D) → (B, T, H, D)
        # .contiguous()：transpose 后内存不连续，view 前必须先变连续（否则报错）
        # .view(B, T, C)：(B, T, H, D) → (B, T, H*D) = (B, T, C)，拼接所有头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影：把拼接后的多头表示映射回 n_embd 维。
        y = self.proj(y)

        # ─── 更新 KV Cache（存储紧凑的 Hk 头，而非扩展后的 H 头）───
        # 注意：缓存存的是旋转后的 K（已含位置信息），下次直接拼接即可，无需再旋转。
        # 语法：torch.cat([kv_cache.k, k], dim=2) 把旧缓存和新 K 在时间维拼接，
        #       k 是本次新 token 对应的 K（未经 GQA 扩展的紧凑版）。
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B, Hk, *, D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        # 把新的 K/V 包装成 KVCache 对象返回，供下一个生成步骤使用。
        new_cache = KVCache(k_new, v_new)
        return y, new_cache

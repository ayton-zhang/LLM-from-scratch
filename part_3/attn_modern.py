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
from kv_cache import KVCache, RollingKV  # 简单缓存 + 滚动缓冲区（支持滑动窗口+sink）

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
        # 设为整数 W 时每个 token 只看最近 W 个历史 token
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
        # 两种缓存类型：
        #   RollingKV（sliding_window 不为 None 时）：step() 自动拼接 + 裁剪到 sink+window，
        #       返回的 k_all/v_all 已经是被裁剪后的局部 K/V，显存固定不增长。
        #   KVCache（sliding_window 为 None 时）：简单拼接，不做裁剪，适用于全局注意力。
        if kv_cache is not None:
            if isinstance(kv_cache, RollingKV):
                # RollingKV.step() 内部自动完成：拼接新 K/V → 裁剪到 sink+window → 返回裁剪后的值
                k_all, v_all = kv_cache.step(k, v)  # (B, Hk, ≤sink+window, D)
            else:
                # KVCache：手动在时间维拼接新旧 K/V，不裁剪
                k_all = torch.cat([kv_cache.k, k], dim=2)  # (B, Hk, Tpast+T, D)
                v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # ─── 滑动窗口 + 注意力水槽裁剪 ───
        # 三种情况分别处理：
        #   1. RollingKV 已维护窗口 → 无需额外裁剪或 mask（缓存已限长）
        #   2. KVCache + sliding_window → 手动裁剪 k_all/v_all（历史总是受限，但缓存未限）
        #   3. 无缓存 + sliding_window → 构造自定义 mask（因果 + 滑窗），不直接裁剪张量
        attn_mask = None
        is_causal = (kv_cache is None) and (self.sliding_window is None)

        if self.sliding_window is not None:
            if isinstance(kv_cache, RollingKV):
                # RollingKV 已自动维护 sink+window，缓存值即最终窗口值，无需额外处理
                pass
            elif kv_cache is not None:
                # KVCache + sliding_window：手动裁剪 K/V，只保留前 sink 个 + 后 window 个
                limit = self.sliding_window + self.attention_sink
                if k_all.size(2) > limit:
                    s = self.attention_sink
                    # 语法：[:, :, -self.sliding_window:, :] 负索引取时间维最后 sliding_window 个
                    k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
                    v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)
            else:
                # ─── 无缓存路径 + 滑动窗口：构造自定义注意力 mask ───
                # 训练 / generate_nocache() 走这个分支。
                # 注意这里只用纯滑动窗口（因果 + 局部距离约束），
                # 不使用 attention_sink —— attention_sink 是推理时 RollingKV
                # 的缓存管理策略（StreamingLLM），不是注意力计算本身的规则。
                # mask[i, j] = -inf 当 (j > i) 或 (j < i - sliding_window + 1)
                T_q = q.size(2)
                T_k = k_all.size(2)
                # 语法：torch.arange(N) 生成 [0,1,...,N-1]，unsqueeze 扩张维度用于广播比较
                row = torch.arange(T_q, device=q.device).unsqueeze(1)  # (T_q, 1)
                col = torch.arange(T_k, device=q.device).unsqueeze(0)  # (1, T_k)
                # 因果约束：j > i → 遮蔽
                causal_mask = col > row
                # 滑窗约束：j < i - W + 1 → 超出局部窗口的旧 token 被遮蔽
                outside_window = col < row - self.sliding_window + 1
                # 语法：torch.where(条件, A, B) 条件为 True 取 A（-inf 遮蔽），否则取 B（0 可见）
                attn_mask = torch.where(causal_mask | outside_window, float('-inf'), 0.0)
                # SDPA 要求 mask 形状 (B, 1, T_q, T_k) 或可广播到该形状
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)

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
        # is_causal：无滑动窗口 + 无缓存时开启因果掩码；有滑动窗口时走自定义 mask。
        # 注意：is_causal=True 与 attn_mask 不能同时传入（PyTorch 会报错），
        #       这里的逻辑保证了它们互斥。
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=attn_mask,
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
        # 缓存存的是旋转后的 K（已含位置信息），下次直接读取即可，无需再旋转。
        # 根据是否开启滑动窗口选择不同的缓存容器：
        #   RollingKV：滑动窗口模式下，step() 自动维护 sink+window 裁剪，显存固定
        #   KVCache：全局注意力模式，简单拼接不裁剪，适用于短序列推理
        if kv_cache is not None:
            if isinstance(kv_cache, RollingKV):
                # RollingKV 已在 step() 中原地更新，直接返回自身即可
                new_cache = kv_cache
            else:
                # KVCache：手动拼接新旧 K/V，包装为新缓存对象
                k_new = torch.cat([kv_cache.k, k], dim=2)  # (B, Hk, *, D)
                v_new = torch.cat([kv_cache.v, v], dim=2)
                new_cache = KVCache(k_new, v_new)
        else:
            # 训练 / 无缓存前向（如 generate_nocache）：
            # 无论是否开启滑动窗口，都用简单 KVCache 包装当前 K/V。
            # RollingKV 是推理时的缓存管理策略（维护 sink+window 裁剪），
            # 训练时缓存不会被后续步骤复用，无需滚动裁剪。
            new_cache = KVCache(k, v)
        return y, new_cache

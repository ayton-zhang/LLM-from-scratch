# ==========================================
# 组件：CausalSelfAttentionModern —— 现代因果自注意力
# ==========================================
# 与 Part 2 的 CausalSelfAttention 相比，本模块做了四处关键升级：
#   1. RoPE（旋转位置编码）替换学习型 pos_emb：位置信息编码进 Q/K 旋转角度，外推性更好
#   2. KV Cache：推理时缓存历史 K/V，每步只算新 token，计算量从 O(T²) 降至 O(T)
#   3. 滑动窗口注意力（Sliding Window）：每个 token 只看最近 W 个位置，显存更省
#   4. GQA（分组查询注意力）：多个 Query 头共享同一对 K/V 头，速度更快、显存更少
#
# 数据流动总览（以推理时的 forward 为例）：
#   x (B,T,C)
#   → wq/wk/wv 线性投影 → Q (B,n_head,T,D), K/V (B,n_kv_head,T,D)
#   → RoPE 旋转 Q/K
#   → 拼接历史 KV Cache（如果有）
#   → 滑动窗口裁剪（如果开了 sliding_window）
#   → GQA 扩展 K/V 头数
#   → F.scaled_dot_product_attention（融合算子，内部做 scale+softmax+加权求和）
#   → 转置+拼接多头 → proj 投影 → y (B,T,C)
#   → 更新 KV Cache → 返回 (y, new_cache)
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class

class CausalSelfAttentionModern(nn.Module):
    # ==========================================
    # 参数初始化
    # ==========================================
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):  # ← NEW
        # 参数说明：
        #   n_embd         : 隐藏层维度（每个 token 用多少个数字表示）
        #   n_head         : Query 注意力头数，决定模型从多少个"视角"关注序列
        #   dropout        : 随机丢弃率，训练时防止过拟合，推理时自动关闭
        #   rope           : True → 用 RoPE 旋转位置编码；False → 无位置信息
        #   max_pos        : RoPE 预计算的最大序列长度，影响外推能力（通常远大于 block_size）
        #   sliding_window : 局部注意力窗口大小（None = 全局注意力）。
        #                    设为整数 W 时，每个 token 只关注最近 W 个位置，
        #                    大幅降低长序列的显存消耗（Mistral 等模型的做法）。
        #   attention_sink : "注意力水槽"——强制保留最开头的 K 个 token 在注意力视野中。
        #                    即使开了 sliding_window，这些"锚点" token 也不会被滑出窗口，
        #                    防止模型在极长序列里完全"忘记"开头的重要信息（StreamingLLM 技术）。
        #   n_kv_head      : KV 头数，用于分组查询注意力 (GQA/MQA)。
        #                    None = 与 n_head 相同（标准 MHA）。
        #                    设为比 n_head 小的数（如 n_kv_head=2, n_head=8）时，
        #                    多个 Query 头共享同一对 K/V 头，显存减少、速度更快（LLaMA-3 采用）。
        super().__init__()
        # 断言 n_embd 必须能被 n_head 整除，否则分头时维度不对。
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        # n_head：Query 的头数，决定模型从多少个"视角"关注序列。
        self.n_head = n_head
        # n_kv_head：K/V 的头数，用于 GQA（分组查询注意力）。
        # 语法：`n_kv_head or n_head` 等价于 `n_kv_head if n_kv_head else n_head`，
        #       当 n_kv_head=None（未指定）时退回到标准 MHA（K/V 头数 = Q 头数）。
        # GQA 直觉：把 8 个 Q 头分成 2 组，每组 4 个 Q 头共享同 1 对 K/V，
        #           K/V 显存减少 4 倍，推理速度更快（LLaMA-3 / Mistral 采用）。
        self.n_kv_head = n_kv_head or n_head      # ← NEW (GQA defaults to MHA)
        # 第二个断言确保 Q 头数能被 K/V 头数整除，否则分组不均匀。
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        # group_size：每组有多少个 Q 头共享同一对 K/V 头。
        # 标准 MHA：group_size=1；MQA（多查询注意力）：group_size=n_head。
        self.group_size = self.n_head // self.n_kv_head
        # d_head：每个注意力头的特征维度，n_embd 均分给所有 Q 头。
        self.d_head = n_embd // n_head

        # GQA 导致 Q 与 K/V 的投影矩阵大小不同，因此分开定义三个线性层。  ← CHANGED
        # wq 输出 n_head * d_head 维（所有 Q 头），
        # wk/wv 只输出 n_kv_head * d_head 维（更少的 K/V 头），节省参数和显存。
        # bias=False：现代大模型普遍去掉线性层偏置，减少参数量且效果相当。
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        # proj：输出投影，把多头拼接后的结果映射回 n_embd 维。
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # Dropout 层：训练时以 dropout 概率随机将注意力权重置零，
        # 类比"随机打晕一些注意力连接"，迫使模型不过度依赖某几个特定位置。
        # 推理时（model.eval()）Dropout 自动关闭。
        self.dropout = nn.Dropout(dropout)

        # use_rope：是否启用旋转位置编码。
        self.use_rope = rope
        # rope_cache 延迟初始化（第一次 forward 时才建立），
        # 因为此时才能确定 device（CPU 还是 GPU），确保 cos/sin 张量在正确的设备上。
        # 语法：`RoPECache | None` 是 Python 3.10+ 的类型提示联合语法，
        #       表示这个属性可以是 RoPECache 或 None。
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

    # ==========================================
    # RoPE 延迟初始化
    # ==========================================
    def _maybe_init_rope(self, device):
        # 延迟初始化 RoPECache：只在第一次 forward 时创建，确保 cos/sin 表与模型在同一 device 上。
        # 为什么不在 __init__ 里创建？因为 __init__ 时还不知道模型在 CPU 还是 GPU 上，
        # 如果提前创建了 CPU 上的张量，后面模型搬到 GPU 时还要手动迁移，不如延迟到第一次 forward 再建。
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    # ==========================================
    # 前向传播
    # ==========================================
    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        # 语法：B, T, C = x.shape 是元组解包，同时获取批大小、序列长度、隐藏维度。
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # ─── 第一步：Q/K/V 投影 ───
        # 把输入的隐藏向量 x 分别投影到 Query、Key、Value 空间。
        # 直觉类比：你在图书馆找书，
        #   Q = "我想找什么主题"（查询意图）
        #   K = "这本书讲什么"（内容标签）
        #   V = "这本书的具体内容"（实际信息）
        # Q 与 K 的点积算出"这本书跟我的兴趣有多匹配"（注意力分数），
        # 然后用这些分数对 V 做加权平均，得到"最相关信息的混合"。
        #
        # 形状变换：
        #   wq(x): (B,T,C) → (B,T, n_head*d_head)
        #   .view(B,T, n_head, d_head): 拆成多个头，(B,T, H, D)
        #   .transpose(1,2): 把头维度提前，(B, H, T, D)，方便后面批量矩阵乘法
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # ─── 第二步：RoPE 旋转位置编码 ───
        # 只对"当前这批新 token"做旋转，缓存里的历史 K 在被存入时已经旋转过了，不能重复旋转。
        # start_pos 指定当前 token 在完整序列中的起始位置，确保旋转角度与绝对位置对应。
        # 例：start_pos=2, T=1 → 当前 token 是序列第 3 个位置，角度 = θ_base * 2。
        #
        # RoPE 的核心思想：把 Q 和 K 的每一对相邻维度当作二维平面上的一个点，
        # 按位置角度旋转，旋转后 Q·K 的点积值自然包含了 (pos_q - pos_k) 的相对距离信息。
        # 这比学习型位置嵌入更优雅——模型不需要背"位置 3 = 向量 [0.1, 0.5, ...]"，
        # 而是从几何旋转中直接推导相对位置。
        # RoPE on *current* tokens (cached keys are already rotated)
        if self.use_rope:
            # torch.arange(start_pos, start_pos + T) 生成当前 token 的位置下标序列。
            # 例：start_pos=0, T=5 → pos=[0,1,2,3,4]，表示 prompt 的前 5 个 token。
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            # rope_cache.get(pos) 从预计算表里按位置取出 cos/sin 值：
            #   cos/sin 形状：(T, d_head/2)，每行对应一个 token 位置的所有频率维度。
            cos, sin = self.rope_cache.get(pos)
            # apply_rope_single 对 Q/K 做实际的旋转操作：
            #   把每个头的特征向量两两配对，按对应位置的角度旋转。
            # 注意 V 不做旋转——V 只存语义内容，位置信息只需编码进"谁关注谁"的打分里。
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)  ← 形状不变，值被旋转
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # ─── 第三步：拼接历史 KV Cache ───
        # 推理时的核心优化：之前算过的 K/V 不要扔，拼接到末尾。
        # 这样每步只需给当前 token 的 Q 与"完整历史 K 做点积"即可，
        # 无需重新计算所有历史 token 的 K/V。
        # 类比：你一边听演讲一边做笔记，不用每听到一句话就回头重读整个讲稿，
        #       只需把新信息追加到笔记末尾，回顾时翻笔记就行。
        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            # 语法：torch.cat([old, new], dim=2) 在时间维（dim=2）拼接。
            # kv_cache.k 形状 (B, Hk, T_past, D) → 拼完后 (B, Hk, T_past+T, D)
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            # 无缓存时（训练或第一次前向），直接用当前 K/V。
            k_all, v_all = k, v

        # ─── 第四步：滑动窗口 + 注意力水槽裁剪 ───
        # 如果总 token 数超过了 sink + window，裁剪中间部分，只保留：
        #   - 前 attention_sink 个"锚点" token（永不丢弃）
        #   - 后 sliding_window 个"最近窗口" token
        # 中间被裁掉的旧 token 信息永久丢失（缓存显存固定的代价）。
        #
        # 为什么需要 attention_sink？StreamingLLM 论文发现：LLM 会自发把大量注意力分数
        # "倾倒"到序列开头的几个 token 上（称为 attention sink）。如果把这些 token 也裁掉，
        # 模型的注意力分布会急剧恶化（perplexity 飙升）。因此即使开滑动窗口，
        # 也要保留最开头的几个 token 作为"注意力垃圾桶"。
        #
        # 训练时的行为：如果序列长度 T > sink+window，也会触发裁剪。
        # 此时 K/V 被裁短（如从 T=8 裁到 4），后续使用 is_causal=True，
        # 注意：裁剪后 K/V 的"绝对位置"变了（原来位置 4-7 变成 0-3），
        # 而 causal mask 认为它们从 0 开始——这对训练有轻微影响，
        # 实际应用中滑动窗口通常在训练时通过自定义 mask 实现，而非直接裁剪。
        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            # s：sink（水槽）大小，即强制保留的开头 token 数。
            s = self.attention_sink
            # 保留两部分：
            #   1. k_all[:, :, :s, :] —— 前 s 个 token（sink 锚点，永不丢弃）
            #   2. k_all[:, :, -self.sliding_window:, :] —— 最后 window 个 token（最近上下文）
            # 中间部分 k_all[:, :, s:-self.sliding_window, :] 被丢弃。
            # 语法：[:, :, -W:, :] 负索引，取时间维最后 W 个；[:, :, :s, :] 取前 s 个。
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # ─── 第五步：GQA 扩展——把 K/V 头复制到与 Q 头数对齐 ───
        # 语法：.repeat_interleave(group_size, dim=1) 沿头维度（dim=1）将每个 K/V 头
        #       重复 group_size 次（交错复制，而非整体拼接），
        #       例如 [k0, k1] 扩展为 [k0, k0, k0, k0, k1, k1, k1, k1]（group_size=4）。
        # 标准 MHA（n_kv_head == n_head）跳过此步，直接使用原始 K/V。
        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H, Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H, Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # ─── 第六步：缩放点积注意力（整个模块的核心）───
        # F.scaled_dot_product_attention 是 PyTorch 2.0+ 的融合算子，
        # 内部自动完成 scale（除以 √d_head）、softmax、dropout、与 V 的加权求和，
        # 比手写拆开算更快（支持 Flash Attention 等底层 CUDA kernel 优化）。
        #
        # 为什么需要 scale = 1/√d_head？
        #   Q·K 的点积值会随 d_head 增大而变大（更多项相加），
        #   导致 softmax 后的分布过于尖锐（接近 one-hot），梯度消失。
        #   除以 √d_head 把点积值的方差压回 1，让 softmax 分布保持"适度柔软"。
        #
        # is_causal=True（无缓存时）：PyTorch 自动创建下三角 mask，
        #   确保每个 token 只能看到它自己和之前的 token（左→右的自回归约束）。
        # is_causal=False（有缓存时）：缓存中的 K/V 已经是历史 token，无需因果 mask，
        #   当前 token 应该能看到所有缓存的位置。
        # Scaled dot-product attention (PyTorch scales internally)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)

        # ─── 第七步：多头拼接 + 输出投影 ───
        # 语法：.transpose(1, 2) 把头维度放回时间维后面：(B, H, T, D) → (B, T, H, D)
        # .contiguous()：transpose 后内存布局不连续，view 前必须先 contiguous 否则报错。
        #   transpose 只交换了 strides（步长），没移动实际数据，
        #   view 要求连续内存，.contiguous() 会拷贝一份真正连续的数据。
        # .view(B, T, C)：把所有头拼接成一个 (B, T, H*D) = (B, T, C) 的张量。
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # proj 投影：把拼接后的多头表示做一次线性变换，让不同头的信息互相融合。
        y = self.proj(y)

        # ─── 第八步：更新 KV Cache ───
        # 缓存存的是旋转后的 K（已含位置信息），下次直接读取即可，无需再旋转。
        # 用紧凑的 Hk 头存储（而非 GQA 扩展后的 H 头），节省显存。
        # 训练时（kv_cache=None）：k_new/v_new 就是当前批次的 K/V，包装成 KVCache 返回。
        # 推理时（kv_cache 不为 None）：把新旧 K/V 拼接为完整历史，存入新的 KVCache。
        # Update KV cache (store compact Hk heads, not expanded)
        if kv_cache is not None:
            # 语法：torch.cat([old, new], dim=2) 在时间维度追加。
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        # KVCache 是来自 kv_cache.py 的 @dataclass 数据容器，
        # 只负责持有 k/v 两个张量，不做任何裁剪逻辑。
        new_cache = KVCache(k_new, v_new)
        # 返回两个值：注意力输出（用于传给下一层或残差连接）、更新后的缓存。
        return y, new_cache

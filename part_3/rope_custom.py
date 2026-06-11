# ==========================================
# 组件：RoPE（旋转位置编码）—— 缓存表 + 旋转函数
# ==========================================
# RoPE (Rotary Position Embedding) 是现代大模型的主流位置编码方案，
# 被 LLaMA、Mistral、Qwen、Gemma 等几乎所有开源模型采用。
#
# 核心思想（用几何直觉理解）：
#   把注意力头内的 d_head 维特征两两配对（如 (d0,d1), (d2,d3), ...），
#   每对看作二维平面上的一个点，按"该对专属的频率 × 该 token 的绝对位置"
#   这个角度旋转该点。旋转后，两个 token 的 Q 和 K 做点积时，
#   点积值自动包含相对位置信息（pos_q - pos_k），而无需学习位置嵌入表。
#
# 类比：想象一个时钟，每个 token 是一个指针，token 位置越靠后，指针转得越远。
#       指针有快有慢（不同维度的频率不同）：秒针转得快（高频，但对位置不敏感），
#       时针转得慢（低频，但对位置很敏感）。RoPE 用多根"指针"的组合编码位置。
#
# 本文件包含两个部分：
#   1. RoPECache：预计算 cos/sin 查找表（懒初始化 + 自动扩展）
#   2. apply_rope_single：对 Q/K 张量施加旋转操作
from __future__ import annotations
import torch
import math

# ==========================================
# RoPECache：预计算旋转角度的 cos/sin 查找表
# ==========================================
# 为什么需要缓存？每次 forward 都实时算 cos/sin 太慢。
# 预先算好一张表，用位置下标直接查表取 cos/sin，速度提升数倍。
#
# 设计要点：
#   - 懒初始化：第一次 forward 时才建表（而非 __init__ 时），
#     因为 __init__ 时还不知道模型在 CPU 还是 GPU 上
#   - 自动扩展：如果 query 的位置超出当前表的最大值，
#     自动扩展表到更大尺寸（通常翻倍），避免频繁重建
class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    # ==========================================
    # 初始化：计算频率表
    # ==========================================
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        # 参数说明：
        #   head_dim : 每个注意力头的维度，必须为偶数（因为要两两配对旋转）
        #   max_pos  : 预计算的最大序列位置（RoPE 支持的最长序列）
        #   base     : 频率基数，默认 10000.0（LLaMA 的标准值）。
        #              更大的 base（如 500000）让高频衰减更慢，适合更长的序列外推。
        #   device   : 张量所在设备（CPU/GPU），用于确保 cos/sin 表在正确设备上
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        # 立即建立 cos/sin 查找表
        self._build(max_pos)

    # ==========================================
    # get()：按位置索引取出对应的 cos/sin 值
    # ==========================================
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        # 返回值：cos, sin 各形状 (T, D/2)，每行对应一个位置的所有频率维度的 cos/sin 值
        #
        # 调用方（attn_modern.py）：
        #   pos = torch.arange(start_pos, start_pos + T)  → positions
        #   cos, sin = self.rope_cache.get(pos)           → 取出对应行的旋转参数
        #
        # 语法：positions.dim() 返回张量的维度数。
        # 如果传入 (1, T) 的二维张量，取第一行转为 (T,) 的一维张量。
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]

        # 检查是否需要扩展表：如果请求的最大位置超过当前表大小，
        # 自动扩展（翻倍或按需），避免因序列变长而报错。
        # 语法：positions.max().item() 取张量最大值转为 Python 标量，+1 是因为位置从 0 开始计数。
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            # 扩展策略：取 need 和 2*max_pos 的较大值，既满足当前需求又留有富余
            self._build(max(need, int(self.max_pos * 2)))

        # 用位置下标在预计算表中索引：
        # positions 如 [2, 3, 4] → cos[positions] 取出第 2,3,4 行的 cos 值
        # 结果形状：(T, D/2)，每行是一个 token 的所有频率的 cos/sin
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin

    # ==========================================
    # _build()：建立 cos/sin 查找表
    # ==========================================
    # 这是 RoPE 的数学核心。下面逐步推导：
    #
    # 第 1 步——计算反频率 inv_freq：
    #   对于头维度中的第 i 对 (i = 0, 2, 4, ..., d_head-2)：
    #     inv_freq[i//2] = 1 / (base ^ (i / d_head))
    #   = 1 / (10000 ^ (i / d_head))
    #
    #   直觉：i 越小（靠近前面的维度对），频率越高（旋转越快），
    #         对相邻位置的区分度越好（位置 5 和 6 在这些维度上差别大）。
    #         i 越大（靠近后面的维度对），频率越低（旋转越慢），
    #         对远距离位置关系更敏感（能感知位置 0 和 100 的差别）。
    #   这就像"秒针+分针+时针"的组合——多个频率共同编码位置，
    #   既能区分相邻 token，也能感知远距离的顺序关系。
    #
    # 第 2 步——计算频率矩阵 freqs：
    #   freqs[pos, i] = pos * inv_freq[i]  （外积）
    #   每个位置的每个维度对都有一个唯一的旋转角度。
    #
    # 第 3 步——取 cos/sin：
    #   对 freqs 矩阵中每个角度取 cos 和 sin，存入查找表。
    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos

        # inv_freq 形状 (D/2,)，每个值对应一个维度对的"旋转频率"。
        # 语法：torch.arange(0, head_dim, 2) 生成 [0, 2, 4, ..., head_dim-2]，
        #       步长 2 是因为每两个相邻维度配成一对。
        #       .float() 把整数转浮点数（后面要做除法）。
        #       / head_dim 做归一化，确保频率在合理范围内。
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))

        # t 形状 (max_pos,)，值是 [0, 1, 2, ..., max_pos-1]，即所有可能的位置下标。
        t = torch.arange(max_pos, device=self.device).float()

        # 语法：torch.outer(a, b) 计算外积（outer product）。
        # a 形状 (N,), b 形状 (M,) → 结果形状 (N, M)。
        # 这里 t (max_pos,) × inv_freq (D/2,) → freqs (max_pos, D/2)。
        # freqs[i, j] = t[i] * inv_freq[j]，即"位置 i 在第 j 个频率维度上的旋转角度"。
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)

        # 分别取 cos 和 sin，建立两张查找表。
        # 每张表形状 (max_pos, D/2)，行号=位置，列号=频率维度。
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)


# ==========================================
# apply_rope_single：对 Q/K 张量施加旋转
# ==========================================
# 数学原理（二维旋转矩阵）：
#   给定一对相邻维度 (x1, x2) 和旋转角度 θ：
#     x1' = x1 * cos(θ) - x2 * sin(θ)
#     x2' = x1 * sin(θ) + x2 * cos(θ)
#   这等价于把向量 (x1, x2) 在二维平面上逆时针旋转 θ 弧度。
#
# 对 Q 和 K 各自旋转后，点积 Q·K 的值变为：
#   Q[0]·K[5]（位置 0 查位置 5 的内容）= 原始语义相似度 * cos(θ_0 - θ_5) + ...
#   即点积自动包含了"位置差 5"的信息——这就是 RoPE 的精妙之处！
#
# 为什么只旋转 Q 和 K，不旋转 V？
#   V 存的是"语义内容"，位置信息只影响"谁关注谁"（由 Q·K 决定），
#   不影响"被关注后取什么内容"（V 的职责）。旋转 V 不会带来额外好处。
def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    # 断言最后一维（d_head）必须是偶数，才能两两配对。
    assert x.size(-1) % 2 == 0

    # 语法：.unsqueeze(0) 在第 0 维前插入一个大小为 1 的新维度。
    # cos/sin 原本形状 (T, D/2)，插入两次后变 (1, 1, T, D/2)，
    # 这样才能与 x 的 (B, H, T, D) 做逐元素广播运算。
    # 广播规则：dim=0 和 dim=1 上 cos 大小为 1，自动扩展到 B 和 H，
    #           dim=2 和 dim=3 上大小匹配 T 和 D/2（一对旋转参数管两个相邻维度）。
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 语法：x[..., ::2] 是高级索引。
    # `...` 表示"前面所有维度保持不变"（即 B, H, T 都保留）。
    # `::2` 表示在最后一维上每隔一个取一个，步长 2。
    # 效果：x 形状 (B, H, T, D) → x1 形状 (B, H, T, D/2)，取偶数位 (d0, d2, d4, ...)。
    # 同理 `1::2` 取奇数位 (d1, d3, d5, ...)。
    x1 = x[..., ::2]   # 偶数位（每对的第一个元素）
    x2 = x[..., 1::2]  # 奇数位（每对的第二个元素）

    # 二维旋转变换（对每一对维度独立旋转）：
    #   x1' = x1 * cos - x2 * sin    （新偶数位）
    #   x2' = x1 * sin + x2 * cos    （新奇数位）
    # 这里 x1, x2, cos, sin 形状都是 (B, H, T, D/2)，逐元素运算（广播）。
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos

    # 把旋转后的奇偶位交错拼回原形状 (B, H, T, D)：
    # 语法：torch.empty_like(x) 创建一个与 x 同形状的未初始化张量，
    #       比 torch.zeros_like 稍快（省去清零步骤，因为我们会全部覆盖）。
    # 然后分别填入偶数位（xr1）和奇数位（xr2）。
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out

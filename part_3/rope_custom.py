# ==========================================
# 组件：RoPE —— 旋转位置编码（Rotary Position Embedding）
# ==========================================
# 与 Part 2 的学习型位置嵌入（pos_emb）相比，RoPE 的核心思想是：
#   不把位置信息"加"到词向量上，而是把它"编码进 Q/K 的旋转角度"里。
# 直觉：把每个 token 的 Q/K 向量想象成二维平面上的箭头，
#       RoPE 根据该 token 的位置，把箭头旋转对应的角度。
#       两个 token 的注意力得分（Q·K 点积）就自然包含了它们的相对位置信息。
# 优势：
#   1. 外推性好 —— 训练时见过位置 0~2047，推理时也能泛化到更长序列
#   2. 无额外参数 —— 旋转角度由公式确定，不需要像 pos_emb 那样学习一张位置查找表
#   3. 相对位置不变性 —— 两个 token 的点积只依赖它们的距离，与绝对位置无关
# LLaMA / Mistral / GPT-NeoX 等现代模型均采用 RoPE。
from __future__ import annotations
import torch
import math

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        # head_dim 必须是偶数，因为 RoPE 把特征维度两两配对，分别做二维旋转。
        # 若 head_dim=64，则有 32 对 (x1, x2)，每对对应一个旋转频率。
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        # base 是频率基底，默认 10000（与原始 Transformer 的位置编码基底相同）。
        # base 越大，不同维度的旋转频率差异越大，模型能区分的最大相对距离越远。
        self.base = base
        self.device = device
        # 预计算 cos/sin 查找表，避免每次 forward 重复计算三角函数（计算量较大）。
        self._build(max_pos)

    def get(self, positions: torch.Tensor):
        # positions: (T,) 或 (1, T)，存储当前批次每个 token 的绝对位置下标。
        # 语法：positions.dim() 返回张量的维度数（rank），
        #       dim()==2 说明是 (1,T) 形状，需要去掉批次维度变成 (T,) 才能作为索引。
        # 所以 _build() 预计算的是"旋转角度的三角函数值"，
        if positions.dim() == 2:
            positions = positions[0]

        # 检查是否需要扩展查找表：
        # positions.max().item() 取出位置下标的最大值（Python 标量），
        # +1 是因为下标从 0 开始，表大小 = 最大下标 + 1。
        # numel() == 0 处理空序列的边界情况，避免 max() 报错。
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # 序列长度超出预计算范围时，动态扩容（至少翻倍，减少频繁重建的开销）。
            self._build(max(need, int(self.max_pos * 2)))

        # 用位置下标直接索引查找表，取出对应行的 cos/sin 值。
        # self.cos 形状 (max_pos, head_dim/2)，用 (T,) 下标索引后得到 (T, head_dim/2)。
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin

    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos

        # 计算每对特征维度对应的旋转频率（逆频率），对应公式里的mθ。
        # 公式：inv_freq_i = 1 / (base ^ (2i / head_dim))，i = 0, 1, ..., head_dim/2 - 1
        # 低维度（i 小）→ 频率高，旋转快，捕捉近距离位置差异；
        # 高维度（i 大）→ 频率低，旋转慢，捕捉远距离位置差异。
        # 类比：像时钟的秒针（快）和时针（慢）分别编码不同粒度的时间信息。
        # 语法：torch.arange(0, head_dim, 2) 生成 [0, 2, 4, ..., head_dim-2] 的整数序列，
        #       步长为 2 是因为每次取一对特征维度（偶数索引）。
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))

        # 生成位置下标序列 [0, 1, 2, ..., max_pos-1]，对应序列中每个可能的位置。
        t = torch.arange(max_pos, device=self.device).float()

        # 语法：torch.outer(a, b) 计算向量外积，结果形状 = (len(a), len(b))。
        # 这里 t 形状 (max_pos,)，inv_freq 形状 (head_dim/2,)，
        # 外积得到 freqs 形状 (max_pos, head_dim/2)：
        #   freqs[pos, i] = pos * inv_freq[i] = pos / (base ^ (2i/head_dim))
        # 即每个位置、每个频率维度对应的旋转角度θ（弧度）。
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)

        # 预计算 cos(mθ) 和 sin(mθ)，存为查找表，供 apply_rope_single() 直接使用。
        # apply_rope_single() 里的旋转公式是：
        #   x1' = x1 * cos(mθ) - x2 * sin(mθ)
        #   x2' = x1 * sin(mθ) + x2 * cos(mθ)
        # 所以这里只需存 cos/sin，不需要存原始角度 freqs。
        # 推理时直接按位置索引取值，避免重复调用三角函数（计算较慢）。
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)


# ==========================================
# 工具函数：apply_rope_single —— 对单个张量施加 RoPE 旋转
# ==========================================
def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    # 再次确认特征维度为偶数，保证两两配对合法。
    assert x.size(-1) % 2 == 0

    # 广播准备：cos/sin 原形状 (T, D/2)，需要扩展到 (1, 1, T, D/2)，
    # 以便与 x 的 (B, H, T, D/2) 形状进行广播（B 和 H 维度自动复制）。
    # 语法：.unsqueeze(0) 在第 0 维插入一个大小为 1 的新维度，
    #       连续调用两次：(T, D/2) → (1, T, D/2) → (1, 1, T, D/2)。
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 把特征维度按奇偶索引拆成两半：
    # 语法：x[..., ::2] 用省略号 ... 保持前面所有维度不变，
    #       ::2 表示从 0 开始每隔 2 取一个（即偶数索引 0,2,4,...）。
    x1 = x[..., ::2]   # 偶数索引特征，形状 (B, H, T, D/2)
    x2 = x[..., 1::2]  # 奇数索引特征，形状 (B, H, T, D/2)

    # 二维旋转公式（复数乘法）：
    #   [x1']   [cos  -sin] [x1]
    #   [x2'] = [sin   cos] [x2]
    # 即：x1' = x1*cos - x2*sin
    #     x2' = x1*sin + x2*cos
    # 直觉：把每对特征 (x1, x2) 看作二维平面上的一个向量，
    #       按照该 token 位置对应的角度旋转它，不改变向量长度，只改变方向。
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos

    # 把旋转后的两半重新交错放回原来的位置，恢复 (B, H, T, D) 形状。
    # torch.empty_like(x) 创建与 x 形状、dtype、device 完全相同的未初始化张量，
    # 比 torch.zeros_like 更快（不需要填零），因为我们会立即写入所有元素。
    out = torch.empty_like(x)
    out[..., ::2] = xr1   # 偶数索引位置放旋转后的实部
    out[..., 1::2] = xr2  # 奇数索引位置放旋转后的虚部
    return out

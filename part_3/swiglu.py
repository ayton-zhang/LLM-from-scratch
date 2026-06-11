# ==========================================
# 组件：SwiGLU —— 门控前馈网络（Gated FFN）
# ==========================================
# SwiGLU 是现代大模型（LLaMA、Mistral、Gemma）标配的前馈网络，
# 替代了传统 Transformer 的 GELU/ReLU 两层 MLP。
#
# 核心创新：引入"门控机制"——让网络自己决定"哪些信息应该通过，哪些应该过滤掉"。
#
# 数学公式：
#   SwiGLU(x) = (x · W1) ⊙ SiLU(x · W2) · W3
#                     ↑                ↑        ↑
#                  "内容分支"      "门控分支"   "输出投影"
#
#   其中 SiLU(x) = x · sigmoid(x)（也叫 Swish 激活函数），
#   ⊙ 表示逐元素乘法（Hadamard product）。
#
# 与普通 MLP 的对比：
#   普通 MLP（GELU）：  y = GELU(x · W_up) · W_down
#                      一个线性投影 → 激活 → 另一个线性投影
#
#   SwiGLU：           y = (x · W1) ⊙ SiLU(x · W2) · W3
#                      两个并行的线性投影 → 门控乘法 → 输出投影
#
# 直觉类比：普通 MLP 是"一条传送带"（信息流过激活函数），
#          SwiGLU 是"两条传送带 + 一个闸门"——
#            W1 产生"内容"（我想表达什么）
#            W2 产生"门控信号"（这段内容有多重要？0=全过滤，1=全通过）
#            ⊙  把内容和门控逐元素相乘，门控小的维度被"关掉"，大的被"放大"
#            W3 把筛选后的信息投影回原始维度
#
# 为什么更好？门控机制让模型学到"稀疏激活"——不是所有特征维度都需要同时激活，
# 只有门控信号认为重要的维度才被保留。这带来了更强的非线性表达能力，
# 同时参数量约是普通 MLP 的 1.5 倍（两个 W1/W2 而非一个 W_up），
# 换来了显著的效果提升（perplexity 更低）。
#
# 关于 mult 参数的注意事项：
#   普通 MLP 的 mult=4 表示 inner_dim = 4 * dim（如 dim=256 → 中间 1024 维）。
#   SwiGLU 的 mult=4 也表示 inner_dim = 4 * dim，但实际参数量更大——
#   因为有两个 dim→inner 的投影矩阵（W1 和 W2），再加上一个 inner→dim 的 W3。
#   LLaMA 的做法是用 mult = 8/3 ≈ 2.67 来让参数量与 mult=4 的普通 MLP 持平。
#   这里用 mult=4 是简化实现，参数稍多但效果更好。
import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    # ==========================================
    # 初始化：三个线性投影 + SiLU 激活 + Dropout
    # ==========================================
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        # 参数说明：
        #   dim     : 输入/输出维度（= n_embd，隐藏层维度）
        #   mult    : 中间层扩展倍数，inner = mult * dim
        #   dropout : 随机丢弃率，训练时在输出投影后随机置零部分元素
        super().__init__()
        inner = mult * dim  # 中间层维度：信息被"展开"到更高维空间做变换

        # w1："内容分支"的投影矩阵，(dim → inner)。
        # 把输入投影到高维空间，产生"候选内容"。
        self.w1 = nn.Linear(dim, inner, bias=False)

        # w2："门控分支"的投影矩阵，(dim → inner)。
        # 与 w1 输入相同但权重独立，产生"门控信号"——
        # 每个维度的值经过 SiLU 后落在 (0, +∞) 或约 (-0.28, +∞) 范围，
        # 小值/负值趋向 0（门关闭），大正值趋向原值（门打开）。
        self.w2 = nn.Linear(dim, inner, bias=False)

        # w3：输出投影，(inner → dim)。
        # 把门控筛选后的高维信息压缩回原始维度。
        self.w3 = nn.Linear(inner, dim, bias=False)

        # SiLU 激活函数（也叫 Swish）：
        #   SiLU(x) = x * sigmoid(x)
        #   形状像一个"平滑版 ReLU"：
        #     x << 0：输出接近 0（门关闭）
        #     x = 0 ：输出 = 0（过渡点）
        #     x >> 0：输出 ≈ x（门全开，信息直接通过）
        # 比 ReLU 好在处处可导（平滑），比 GELU 好在计算更快（只需 sigmoid）。
        self.act = nn.SiLU()  # SiLU = Swish activation

        # Dropout：训练时随机丢弃输出投影后的部分元素，防止过拟合。
        # 推理时（model.eval()）自动透传，不做丢弃。
        self.drop = nn.Dropout(dropout)

    # ==========================================
    # forward：门控前向传播
    # ==========================================
    def forward(self, x):
        # 输入 x 形状：(B, T, dim)，来自 Block 的 self.ln2(x)。
        # 返回值形状：(B, T, dim)，与输入相同（FFN 的输出加回残差）。

        # ─── 第一步：内容分支投影 ───
        # w1(x) 形状：(B, T, dim) → (B, T, inner)
        # 产生"候选内容"：每个 token 的每个特征维度在 inner 维空间中的表达。
        a = self.w1(x)

        # ─── 第二步：门控分支投影 + 激活 ───
        # w2(x) → SiLU： (B, T, dim) → (B, T, inner) → SiLU → (B, T, inner)
        # 产生"门控信号"：SiLU 后接近 0 的维度被"关门"，
        # 正值大的维度被"开门"，让对应的内容通过。
        b = self.act(self.w2(x))

        # ─── 第三步：门控乘法 + 输出投影 + Dropout ───
        # a * b：逐元素乘法（Hadamard product），形状 (B, T, inner)。
        #   门控信号 b 对内容 a 做"选择性过滤"——
        #   被关门的维度（b≈0）→ 内容被清零
        #   被开门的维度（b≈1+）→ 内容原样通过（或放大）
        #   被半开的维度（0<b<1）→ 内容被衰减
        # w3(...)：输出投影 (B, T, inner) → (B, T, dim)，回到原始维度。
        # Dropout(...)：训练时随机丢弃，推理时透传。
        return self.drop(self.w3(a * b))

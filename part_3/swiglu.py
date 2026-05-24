# ==========================================
# 组件：SwiGLU —— 门控前馈网络
# ==========================================
# 与 Part 2 的经典 FFN（Linear → GELU → Linear）相比，SwiGLU 引入了"门控机制"：
#   经典 FFN : y = W3 · GELU(W1 · x)
#   SwiGLU   : y = W3 · (W1 · x  ⊗  Swish(W2 · x))
#                          ↑ 值分支    ↑ 门控分支
# 直觉：把信息流想象成一扇可调光的灯：
#   - 值分支（W1）产生"原始信息"
#   - 门控分支（W2 + Swish）产生 0~1 之间的"亮度旋钮"
#   - 两者逐元素相乘，让网络自行决定每个维度"放多少信息通过"
# Swish(x) = x · sigmoid(x)，是 SiLU 的别名，比 GELU 更平滑，梯度更稳定。
# LLaMA / PaLM / GPT-4 等现代大模型均采用 SwiGLU 作为 FFN。
import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # inner 是隐藏层维度，默认为输入维度的 4 倍（与经典 FFN 扩张比一致）。
        # 注意：SwiGLU 有两条并行的上投影（w1、w2），参数量约为经典 FFN 的 1.5 倍，
        # 实践中常把 mult 调小到 ~2.67 以保持总参数量不变（LLaMA 的做法）。
        inner = mult * dim

        # w1：值分支的上投影，把 dim 维输入映射到 inner 维"原始信息"。
        # bias=False：现代大模型普遍去掉线性层的偏置，减少参数量且效果相当。
        self.w1 = nn.Linear(dim, inner, bias=False)

        # w2：门控分支的上投影，把 dim 维输入映射到 inner 维"门控信号"。
        # w1 和 w2 接收相同的输入 x，但学习不同的投影方向，分别扮演不同角色。
        self.w2 = nn.Linear(dim, inner, bias=False)

        # w3：下投影，把门控后的 inner 维特征压缩回 dim 维，与残差路径形状对齐。
        self.w3 = nn.Linear(inner, dim, bias=False)

        # SiLU（Sigmoid Linear Unit）即 Swish 激活函数：f(x) = x · σ(x)
        # 与 ReLU 相比：无硬性截断，x<0 时仍有微小梯度，训练更稳定；
        # 与 GELU 相比：计算更简单（无需近似），且经实验在门控结构中效果更好。
        self.act = nn.SiLU()

        # Dropout 施加在最终输出上，随机将部分特征置零，防止过拟合。
        # dropout=0.0 时 nn.Dropout 相当于恒等映射，推理时自动关闭。
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 值分支：线性投影，形状 (B, T, dim) → (B, T, inner)
        a = self.w1(x)

        # 门控分支：线性投影后过 SiLU 激活，产生 0~∞ 的"软门"信号。
        # 形状同样 (B, T, dim) → (B, T, inner)
        b = self.act(self.w2(x))

        # 门控融合：a * b 是逐元素乘法（Hadamard 积），
        # 让门控分支 b 决定值分支 a 中每个维度"保留多少"。
        # 再经 w3 下投影压缩回 dim 维，最后 Dropout 防止过拟合。
        # 形状：(B, T, inner) → (B, T, dim)
        return self.drop(self.w3(a * b))
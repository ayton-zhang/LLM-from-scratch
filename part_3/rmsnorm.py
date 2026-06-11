# ==========================================
# 组件：RMSNorm —— 均方根归一化
# ==========================================
# RMSNorm 是 LayerNorm 的"简化加速版"，被 LLaMA、Mistral、Qwen 等
# 几乎所有现代大模型采用。与 LayerNorm 的核心区别只有一条：
#
#   LayerNorm：y = (x - mean) / std * γ + β    （先去均值，再除标准差）
#   RMSNorm  ：y = x / rms(x) * g              （只除均方根，不中心化）
#
# 为什么去掉 mean 和 bias？
#   1. 更快：少算一次均值 + 少一个可学习参数 β（bias），
#      大模型中节省约 5-10% 的前向时间
#   2. 效果相当：论文和大量实践表明去掉 center 不影响收敛和最终效果
#   3. LLaMA 选择：Meta 的 LLaMA 系列用 RMSNorm，社区跟随验证了它的有效性
#
# 类比：LayerNorm 是"把学生成绩标准化到平均分 0、标准差 1"（做两步），
#       RMSNorm 是"只除以标准差，不管平均分"（做一步，更快）。
#       实际效果差不多，因为 Transformer 的残差连接天然起到了一定的
#       "中心化"作用（x + attn_out 中两部分的均值和方差互相调节）。
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    # ==========================================
    # 初始化：可学习的缩放参数
    # ==========================================
    def __init__(self, dim: int, eps: float = 1e-8):
        # 参数说明：
        #   dim : 归一化的维度（通常 = n_embd，即隐藏层维度）。
        #         RMSNorm 沿最后一维（dim=-1）计算均方根，对每个 token 的
        #         每个特征维度独立归一化。
        #   eps : 防止除零的小常数（1e-8），加在 rms² 里避免 rms=0 时爆炸。
        #         为什么 eps 比 LayerNorm 的默认 1e-5 小？因为 rms 没有
        #         减去均值，数值稳定性更好，可以用更小的 eps。
        super().__init__()
        self.eps = eps

        # weight（论文里叫 g，gain）：可学习的缩放参数，形状 (dim,)。
        # 归一化后每个特征维度乘上对应的 weight，让模型自己决定"这个维度应该放大还是缩小"。
        # 初始化为全 1，即刚开始不做任何缩放，让模型在训练中自行调整。
        #
        # 注意：RMSNorm 没有 bias（β）参数！LayerNorm 有 weight + bias 两个可学习参数，
        # RMSNorm 只有 weight 一个。这是"去掉 center"的体现——不需要偏置来补偿去均值。
        #
        # 语法：nn.Parameter(tensor) 把普通张量包装成"可训练参数"。
        # 与普通张量的区别：nn.Parameter 会被 model.parameters() 遍历到，
        # 从而被优化器自动更新。如果用普通张量存储 weight，优化器会忽略它。
        self.weight = nn.Parameter(torch.ones(dim))

    # ==========================================
    # forward：RMS 归一化
    # ==========================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状：(B, T, D) 或任意多维，沿最后一维（dim=-1）归一化。
        # 返回值形状与输入完全相同。

        # 逐步拆解 RMS 计算：
        #
        #   1. x.pow(2)
        #      逐元素平方。x 中每个值变成 x²，形状不变。
        #
        #   2. .mean(dim=-1, keepdim=True)
        #      沿最后一维求均值。为什么 keepdim=True？
        #       不 keepdim：(B, T, D) → (B, T)，少了一维，无法与原始 x 做除法。
        #       保持维度：(B, T, D) → (B, T, 1)，最后一维大小=1 但维度还在，
        #       广播时自动扩展到 (B, T, D) 与 x 对齐。
        #      这是 PyTorch 中"沿某维做统计后还要与原张量运算"的标准写法。
        #
        #   3. .add(self.eps)
        #      加 eps 防止 sqrt(0) = 0 导致除零。
        #      等价于 x.pow(2).mean(...) + self.eps。
        #
        #   4. .sqrt()
        #      开平方，得到均方根 rms(x) = sqrt(mean(x²) + eps)。
        #      此时 rms 形状 (B, T, 1)，每个位置一个标量表示"这组特征的 RMS 有多强"。
        #
        #   5. x / rms
        #      广播除法：(B, T, D) / (B, T, 1) → (B, T, D)。
        #      把每个 token 的 D 维特征除以它的 RMS，使归一化后 RMS ≈ 1。
        #      直觉：把信号的"音量"调到统一水平，但保留各维度之间的相对比例。
        #
        #   6. * self.weight
        #      广播乘法：(B, T, D) * (D,) → (B, T, D)。
        #      weight 沿最后一维广播，对每个特征维度做可学习的缩放。
        #      让模型自己决定"这个特征维度应该强调还是抑制"。
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

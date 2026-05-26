# ==========================================
# 组件：RMSNorm —— 均方根层归一化
# ==========================================
# 与 Part 2 使用的 LayerNorm 相比，RMSNorm 做了一处关键简化：
#   LayerNorm : y = (x - mean) / std * γ + β   （需要计算均值和方差，有两个可学习参数 γ、β）
#   RMSNorm   : y = x / rms(x) * γ             （只计算均方根，省掉均值偏移，只有一个可学习参数 γ）
# 直觉：LayerNorm 同时做"对齐均值"和"缩放方差"两件事，
#       RMSNorm 认为"对齐均值"不那么重要，去掉它可以减少约 7% 的计算量，且效果几乎不变。
# LLaMA / Mistral / GPT-4 等现代大模型均采用 RMSNorm。
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        # eps（epsilon）是一个极小值，防止分母为零导致数值不稳定。
        # 默认 1e-8，比 LayerNorm 默认的 1e-5 更小，因为 x^2 的量级通常比 x 更大，
        # 分母不容易趋近于零，可以设得更小以减少对归一化结果的干扰。
        self.eps = eps
        # weight（γ）是逐维度的可学习缩放参数，初始化为全 1（即归一化后不做任何缩放）。
        # 相比 LayerNorm，RMSNorm 没有 bias（β）参数，参数量减少一半。
        # 语法：nn.Parameter(torch.ones(dim)) 把一个普通张量包装成"可学习参数"，
        #       PyTorch 会自动把它加入模型的 parameters() 列表，参与梯度更新。
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：计算每个向量的均方根（RMS）
        # x.pow(2)            : 逐元素平方，x² → 形状不变
        # .mean(dim=-1, ...)  : 沿最后一维（特征维）求均值，得到每个 token 的均方值
        # keepdim=True        : 保留被压缩的维度，使形状从 (..., dim) → (..., 1)，
        #                       方便后续广播除法（否则形状对不上）
        # .add(self.eps)      : 加上极小值防止除以零
        # .sqrt()             : 开根号得到均方根值
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # 第二步：归一化 + 缩放
        # x / rms   : 每个 token 的特征向量除以其均方根，将向量"等比例压缩"到单位尺度
        # * self.weight : 逐维度乘以可学习缩放参数 γ，让模型自行决定每个维度的重要程度
        # 广播规则：x 形状 (B, T, dim)，rms 形状 (B, T, 1)，
        #           除法时 1 自动广播到 dim，每个特征维度共享同一个 rms 值。
        return (x / rms) * self.weight
# ==========================================
# 测试：RMSNorm 输出形状不变
# ==========================================
# RMSNorm 归一化只改变向量的"尺度"，不改变张量的"形状"。
# 这个最简测试验证的就是这个基本性质：进去什么形状，出来还是什么形状。

import torch
from rmsnorm import RMSNorm


def test_rmsnorm_shapes():
    # 构造输入：(B=2, T=3, dim=8)——2 个样本，每个样本 3 个 token，每个 token 用 8 维向量表示。
    # torch.randn：从标准正态分布 N(0,1) 随机采样，产生有正有负的值。
    x = torch.randn(2, 3, 8)

    # 创建 RMSNorm 层，dim=8 表示沿最后一维（8 维特征向量）做归一化。
    # 注意：RMSNorm(8) 内部自动创建了一个形状为 (8,) 的可学习参数 weight（初始全 1）。
    rn = RMSNorm(8)

    # forward：x 传入 RMSNorm → 每个 token 的 8 维向量除以其均方根（rms），再乘以 weight。
    y = rn(x)

    # 核心断言：输出形状必须等于输入形状。
    # 归一化操作只在 dim=-1 上做除法和乘法，不会改动 batch、时间、特征这三个维度的大小。
    assert y.shape == x.shape

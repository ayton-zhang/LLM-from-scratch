# ==========================================
# 测试：RMSNorm 数学正确性
# ==========================================
# 验证 RMSNorm 不仅是"形状不变"，而是真正做了归一化：
#   1. 归一化后 RMS ≈ 1（数学定义的正确性）
#   2. weight 参数实际生效
#   3. 反向传播梯度正常流通

import torch
from rmsnorm import RMSNorm


def test_rmsnorm_normalizes_to_unit_rms():
    """RMSNorm 归一化后，沿最后一维的 RMS 应 ≈ 1.0。

    这是 RMSNorm 的核心数学保证：输出向量被缩放到单位尺度，
    后续的 weight 参数再逐维度调整。
    """
    x = torch.randn(4, 8, 16)  # (B=4, T=8, dim=16)
    # weight 全设为 1 → 归一化后不引入额外缩放，便于验证 RMS ≈ 1
    rn = RMSNorm(dim=16)
    rn.weight.data.fill_(1.0)  # 覆盖为全 1，消除随机初始化的影响
    y = rn(x)

    # 手动计算输出沿最后一维的 RMS，公式：rms = sqrt(mean(y²))
    rms_out = y.pow(2).mean(dim=-1).sqrt()  # 形状 (B, T)

    # 每个 token 的 RMS 应接近 1.0
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-5), \
        f"RMS after RMSNorm should be ~1.0, got {rms_out}"


def test_rmsnorm_weight_applied():
    """验证 weight 参数实际参与缩放，而非被忽略。

    故意设置 weight 不全为 1，对比"手动归一化 × weight"的结果
    与 RMSNorm 输出是否一致。
    """
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (B=1, T=1, dim=4)
    rn = RMSNorm(dim=4)

    # 设 weight = [0.5, 1.0, 2.0, 3.0]
    w = torch.tensor([0.5, 1.0, 2.0, 3.0])
    rn.weight.data.copy_(w)

    y = rn(x)  # RMSNorm 输出

    # 手动公式：rms = sqrt(mean(x²)+eps),  y_manual = x / rms * w
    eps = 1e-8
    rms_manual = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_manual = (x / rms_manual) * w.unsqueeze(0).unsqueeze(0)

    assert torch.allclose(y, y_manual, atol=1e-6), \
        f"RMSNorm output should match manual formula. y={y}, y_manual={y_manual}"


def test_rmsnorm_backward():
    """验证 RMSNorm 的反向传播梯度正常流通。

    如果 backward 失败，训练时该层的 weight 无法更新，
    这是一个"防死"测试。
    """
    x = torch.randn(2, 3, 8, requires_grad=True)
    rn = RMSNorm(dim=8)

    y = rn(x)
    # 构造一个简单的损失：所有输出的和（不是真正的训练损失，但足以触发反向传播）
    loss = y.sum()
    loss.backward()

    # 1) weight 的梯度应存在且不含 NaN
    assert rn.weight.grad is not None, "weight.grad should not be None after backward"
    assert not torch.isnan(rn.weight.grad).any(), "weight.grad contains NaN"

    # 2) 输入 x 的梯度应存在（梯度可以流过归一化层）
    assert x.grad is not None, "x.grad should not be None (gradient flows through RMSNorm)"
    assert not torch.isnan(x.grad).any(), "x.grad contains NaN"

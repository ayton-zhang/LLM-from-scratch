# ==========================================
# 单元测试：验证 PolicyWithValue 策略网络前向传播与输出张量形状
# ==========================================

import torch
from policy import PolicyWithValue

def test_policy_shapes():
    # 设定测试超参数：批次大小 B=2，序列长度 T=16，词表大小 V=256
    B, T, V = 2, 16, 256

    # 实例化微型 PolicyWithValue 模型（包含 Actor 策略 LM 与 Critic 价值头 Value Head）
    pol = PolicyWithValue(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=64)

    # 随机生成形状为 (B, T) 的整数输入 Token ID 张量 x
    x = torch.randint(0, V, (B, T))

    # 前向传播计算 logits 分值与 values 状态价值
    logits, values, loss = pol(x, None)

    # 验证断言：
    # 1. logits 形状必须匹配 (B, T, V)
    assert logits.shape == (B, T, V)
    # 2. values 形状必须为 (B, T)（每个时间步输出一个标量价值）
    assert values.shape == (B, T)

# ==========================================
# 测试：RoPE 旋转位置编码 —— 数学正确性 + 注意力集成
# ==========================================
# 验证 RoPE 不只是"把值改了一下"，而是：
#   1. 位置 0 时旋转角度为 0，输出与输入相同（边界正确性）
#   2. 相对位置影响 Q·K 点积（核心功能：位置编码生效）
#   3. RoPE 可以正确集成到 CausalSelfAttentionModern 中（组件协同）

import torch
from rope_custom import RoPECache, apply_rope_single


def test_rope_position_zero_is_identity():
    """位置 0 时 cos(0)=1, sin(0)=0，旋转操作应恒等变换。

    这是 RoPE 的"地基"：如果位置 0 都旋转错了，那整个公式就崩了。
    位置 0 是序列的第一个 token，旋转角度 θ = pos * 频率，
    当 pos=0 时所有频率的 θ=0，cos=1, sin=0 → 旋转矩阵退化为单位矩阵。
    """
    B, H, T, D = 2, 4, 3, 8  # D=8 必须是偶数
    rc = RoPECache(head_dim=D, max_pos=32)
    x = torch.randn(B, H, T, D)

    # 所有 token 都在位置 0 → 旋转角度全为零
    pos = torch.zeros(T, dtype=torch.long)
    cos, sin = rc.get(pos)  # cos 应全为 1，sin 应全为 0

    y = apply_rope_single(x, cos, sin)

    # 旋转角度为 0 时，输出应完全等于输入
    assert torch.allclose(y, x, atol=1e-6), \
        "RoPE at position 0 should be identity (no rotation)"


def test_rope_relative_position_affects_dot_product():
    """同一 Q 与不同位置的 K 做点积，得分应不同。

    这是 RoPE 核心机制的实际验证：
    - Q@pos_p 与 K@pos_p（同位置）点积大 → 自注意力学"关注自己"
    - Q@pos_p 与 K@pos_q（不同位置）点积可能不同 → 相对位置编码生效

    如果没有 RoPE（或 RoPE 实现错误），同一 Q 与所有 K 的点积差异
    会与 RoPE 正确时不同。这里不验证"差多少"，只验证"有差异"。
    """
    D = 16
    # 头数 H=1 简化分析：单一 Q 头对应单一 K 头
    rc = RoPECache(head_dim=D, max_pos=64)

    # 构造简单可预测的 Q/K：全 1 张量
    # 全 1 的好处：旋转前后可直观理解变化
    q = torch.ones(1, 1, 1, D)    # (B=1, H=1, T=1, D=16)
    k = torch.ones(1, 1, 3, D)    # (B=1, H=1, T=3, D=16)

    # Q 在位置 0，K 在位置 [0, 1, 2]
    q_pos = torch.tensor([0])
    k_pos = torch.tensor([0, 1, 2])

    q_cos, q_sin = rc.get(q_pos)
    k_cos, k_sin = rc.get(k_pos)

    q_rot = apply_rope_single(q, q_cos, q_sin)  # (1, 1, 1, 16)
    k_rot = apply_rope_single(k, k_cos, k_sin)  # (1, 1, 3, 16)

    # 计算 Q 与 3 个 K 的点积分数（未除以 √d_head 的原始分数）
    # q_rot 形状 (1,1,1,16)，k_rot 形状 (1,1,3,16)
    # 沿最后一维求和 → (1,1,3)
    scores = (q_rot * k_rot).sum(dim=-1).squeeze()

    assert scores.shape == (3,), f"Expected scores shape (3,), got {scores.shape}"

    # 三个位置的 K 对应的分数应不同（相对位置编码生效）
    # scores[0]: Q@pos0 vs K@pos0, scores[1]: Q@pos0 vs K@pos1, scores[2]: Q@pos0 vs K@pos2
    assert not torch.allclose(scores[0], scores[1], atol=1e-5), \
        f"RoPE should make scores differ by relative position. Got all similar: {scores}"
    assert not torch.allclose(scores[0], scores[2], atol=1e-5), \
        f"RoPE should make scores differ by relative position. Got all similar: {scores}"


def test_rope_with_attention_backward():
    """验证 RoPE + 注意力的组合可以正常进行前向和反向传播。

    构造一个最小 CausalSelfAttentionModern（启用 RoPE），喂入随机 token，
    验证前向输出形状正确、反向传播梯度流通。
    """
    from attn_modern import CausalSelfAttentionModern

    n_embd, n_head, T = 64, 4, 4
    attn = CausalSelfAttentionModern(
        n_embd=n_embd, n_head=n_head, dropout=0.0,
        rope=True, max_pos=64,
    )

    x = torch.randn(2, T, n_embd)
    y, cache = attn(x, kv_cache=None, start_pos=0)

    # 前向输出形状正确
    assert y.shape == x.shape, f"Attention output shape {y.shape} != input {x.shape}"

    # 反向传播：梯度流过 RoPE + 注意力
    loss = y.sum()
    loss.backward()

    # 检查 Q 投影权重的梯度存在
    assert attn.wq.weight.grad is not None, "wq.grad should not be None"
    assert attn.wk.weight.grad is not None, "wk.grad should not be None"
    assert not torch.isnan(attn.wq.weight.grad).any(), "wq.grad contains NaN"
    assert not torch.isnan(attn.wk.weight.grad).any(), "wk.grad contains NaN"

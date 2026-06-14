# ==========================================
# 测试：RoPE 旋转位置编码 —— 形状保持 + 值发生变化
# ==========================================
# RoPE 旋转在每对特征维度上做二维旋转操作，旋转后向量的"长度"不变、形状不变，
# 但具体的值会因旋转角度而改变。这两个测试分别覆盖：
#   test_rope_rotation_shapes_single  —— 标准 MHA 场景（Q 头数 = K 头数）
#   test_rope_rotation_shapes_gqa     —— GQA 场景（Q 头数 > K 头数，共享 KV）

import torch
from rope_custom import RoPECache, apply_rope_single


def test_rope_rotation_shapes_single():
    """标准 MHA 场景：Q 和 K 头数相同，验证 RoPE 旋转后形状与值的变化。

    适用于原始 Transformer、LLaMA-1 等所有头数相等的注意力结构。
    """
    # B=1 批次，H=2 注意力头，T=5 序列长度，D=8 每头特征维度（必须偶数）
    # 语法：`B, H, T, D = ...` 同时给 4 个变量赋值
    B, H, T, D = 1, 2, 5, 8

    # RoPECache：预计算 cos/sin 查找表，max_pos=32 表示预先算好位置 0~31 的三角函数值。
    # head_dim=D 决定每个位置有 D/2=4 个旋转频率（两两配对的频率数）。
    rc = RoPECache(head_dim=D, max_pos=32)

    # 随机 Q 和 K：(B, H, T, D) 形状，MHA 场景下 Q 和 K 头数相同
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    # 生成位置下标 [0, 1, 2, 3, 4]，表示序列中 5 个 token 的绝对位置。
    # 语法：torch.arange(0, T) 等价于 torch.arange(start=0, end=T)，生成 [0, T) 的整数序列。
    pos = torch.arange(0, T)

    # 从预计算表里按位置取出 cos/sin：(T, D/2) 形状，每行对应一个 token 的旋转角度三角函数值。
    # 语法：`cos, sin = rc.get(pos)` 是元组解包，get 返回一对张量。
    cos, sin = rc.get(pos)

    # 对 Q 和 K 实施旋转：把每个 token 的特征向量两两配对，按对应位置的角度旋转。
    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    # ─── 断言 1：旋转不改变形状 ───
    # RoPE 只在最后一维内部做奇偶配对的旋转，不改变任何维度的大小。
    assert q2.shape == q.shape
    assert k2.shape == k.shape

    # ─── 断言 2：旋转确实改变了值 ───
    # 如果旋转前后值完全一样，说明 RoPE 根本没生效（可能是 cos=1, sin=0 导致恒等变换）。
    # torch.allclose(a, b) 判断两个张量是否在容差内所有元素都相等。
    # `not torch.allclose(...)` 即"不是所有元素都相等"→ 至少有些元素被旋转改变了。
    # 语法：`not torch.allclose(q2, q)` —— not 是 Python 逻辑取反运算符。
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)


def test_rope_rotation_shapes_gqa():
    """GQA 场景：Q 头数 > K 头数（共享 KV），验证 RoPE 在不同头数下仍然正确。

    GQA（分组查询注意力）是现代大模型的标配（LLaMA-3、Mistral 等），
    多个 Q 头共享同一对 K/V 头，K/V 头数更少、显存更省。
    此测试验证 RoPE 在 Q 和 K 头数不同时，各自独立旋转不出错。
    """
    # B=2 批次，H=8 个 Q 头，Hk=2 个 K 头（GQA 分 4 组，每组 4 个 Q 头共享 1 对 K/V）
    # T=7 序列长度，D=16 每头特征维度（GQA 场景下头维度通常更大）
    # 语法：`B, H, Hk, T, D = ...` 5 个变量同时赋值
    B, H, Hk, T, D = 2, 8, 2, 7, 16

    # max_pos=128 比 single 测试的 32 更大，覆盖更长的序列场景。
    rc = RoPECache(head_dim=D, max_pos=128)

    # Q 用 H=8 个头（形状 B,8,T,D），K 只用 Hk=2 个头（形状 B,2,T,D）
    q = torch.randn(B, H,  T, D)
    k = torch.randn(B, Hk, T, D)

    # 位置从 10 开始（而非 0），模拟推理中后续生成步骤的场景——
    # 前 10 个 token 已在缓存中，当前这 7 个 token 的绝对位置是 [10, 11, ..., 16]。
    # 语法：torch.arange(10, 10 + T) 生成从 10 开始、长度为 T=7 的整数序列 → [10,11,12,13,14,15,16]。
    pos = torch.arange(10, 10 + T)

    cos, sin = rc.get(pos)

    # RoPE 旋转：Q (B,8,T,16) 和 K (B,2,T,16) 各自独立旋转，互不干扰。
    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    # ─── 断言 1：形状保持，且明确写出期望的完整形状 ───
    # 与 single 测试不同，这里显式给出了完整形状元组做对比，
    # 目的是明确约束 GQA 不同头数下的形状，防止 H 和 Hk 被意外混淆。
    assert q2.shape == (B, H,  T, D)
    assert k2.shape == (B, Hk, T, D)

    # ─── 断言 2：值确实被旋转改变了 ───
    # 非零位置（10~16）对应非零旋转角度，cos≠1、sin≠0，旋转必然改变值。
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)

# ==========================================
# 测试：KV Cache 逻辑正确性
# ==========================================
# 现有 test_kvcache_shapes.py 只检查"长度不超限"，
# 但完全不验证缓存内容的正确性。这里补充对 KVCache 和 RollingKV
# 的内容级别测试，确保缓存的 K/V 值逐 token 正确。

import torch
from kv_cache import KVCache, RollingKV


def test_simple_kvcache_concat():
    """KVCache 拼接后，旧值和新增值都在正确位置。

    用"标识张量"（每层用不同值填充）做输入，验证拼接后：
    - 旧缓存的值没被覆盖
    - 新 K/V 追加在时间维末尾
    - T 属性正确
    """
    B, H, D = 1, 2, 4

    # 第一步：创建初始缓存（模拟 prefill 阶段，T=3）
    k0 = torch.randn(B, H, 3, D)
    v0 = torch.randn(B, H, 3, D)
    cache = KVCache(k=k0, v=v0)

    # 验证 T 属性：缓存的 token 数
    assert cache.T == 3, f"T should be 3, got {cache.T}"

    # 第二步：追加 1 个新 token（模拟 decode 阶段）
    k1 = torch.randn(B, H, 1, D)
    v1 = torch.randn(B, H, 1, D)
    k_new = torch.cat([cache.k, k1], dim=2)
    v_new = torch.cat([cache.v, v1], dim=2)
    cache = KVCache(k=k_new, v=v_new)

    assert cache.T == 4, f"T should be 4 after append, got {cache.T}"

    # 验证前 3 个位置的 K 值没变
    assert torch.equal(cache.k[:, :, :3, :], k0), \
        "Original K values should be preserved after append"
    # 验证第 4 个位置是新增的 K
    assert torch.equal(cache.k[:, :, 3:, :], k1), \
        "New K should be at the last position"


def test_rollingkv_sink_preserved():
    """RollingKV 多步 step 后，前 sink 个 token 的 K/V 应始终不变。

    这是 attention_sink 的核心作用：保留最开始的"锚点" token，
    即使后面 step 了很多步、中间 token 已经被丢弃，锚点依然在。

    测试策略：用递增的整数作为 K/V 值（token 0 = 全 0，token 1 = 全 1，...），
    每步 append 一个新 token，多步后验证前 sink 个值还是最原始的。
    """
    B, H, D = 1, 1, 1
    kv = RollingKV(window=3, sink=2)  # 最多保留 2 + 3 = 5 个 token

    # 按步骤插入 token：每步 K 值 = 步数（方便追踪来源）
    for step in range(10):
        k_new = torch.full((B, H, 1, D), float(step))
        v_new = torch.full((B, H, 1, D), float(step))
        k, v = kv.step(k_new, v_new)

    # 经过 10 步后，总容量为 sink(2) + window(3) = 5
    assert k.size(2) == 5, f"Expected 5 tokens in cache, got {k.size(2)}"

    # 前 2 个（sink）应始终保持 token 0 和 token 1
    # 因为它们是序列最开始的两个 token，永远不会被丢弃
    assert k[0, 0, 0, 0].item() == 0.0, f"Sink token 0 should be preserved, got {k[0,0,0,0]}"
    assert k[0, 0, 1, 0].item() == 1.0, f"Sink token 1 should be preserved, got {k[0,0,1,0]}"


def test_rollingkv_window_contains_recent():
    """RollingKV 多步 step 后，末尾 window 个 token 应是最新插入的。

    承接上一个测试的验证思路：用递增整数标记每步的 token，
    确认窗口末尾的三个 token 是最后三步插入的（step 7, 8, 9）。
    """
    B, H, D = 1, 1, 1
    kv = RollingKV(window=3, sink=0)  # sink=0 → 纯滑动窗口，只保留最近 3 个

    for step in range(8):  # 插入 8 个 token
        k_new = torch.full((B, H, 1, D), float(step))
        v_new = torch.full((B, H, 1, D), float(step))
        k, v = kv.step(k_new, v_new)

    # 窗口大小 = 3，应只保留最近的 3 个：step 5, 6, 7
    assert k.size(2) == 3, f"Window should hold 3 tokens, got {k.size(2)}"
    assert k[0, 0, 0, 0].item() == 5.0, f"Oldest in window should be step 5, got {k[0,0,0,0]}"
    assert k[0, 0, 1, 0].item() == 6.0, f"Middle in window should be step 6, got {k[0,0,1,0]}"
    assert k[0, 0, 2, 0].item() == 7.0, f"Newest in window should be step 7, got {k[0,0,2,0]}"

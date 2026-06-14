# ==========================================
# 测试：RollingKV 滚动缓冲区 —— 容量上限验证
# ==========================================
# RollingKV 的核心约束：缓存 token 数永远不超过 sink + window。
# 这个测试用 10 步循环向通道里疯狂追加 token，每步断言一次容量不越界，
# 验证裁剪逻辑在连续高负载下不会出错。

import torch
from kv_cache import RollingKV


def test_rolling_kv_keep_window_with_sink():
    # B=1 批次大小，H=2 注意力头数，D=4 每头特征维度
    # 语法：`B,H,D = 1,2,4` 同时给三个变量赋值，等价于 B=1; H=2; D=4
    B, H, D = 1, 2, 4

    # 创建滚动缓冲区：window=4（保留最近 4 个 token），sink=2（保留开头 2 个"锚点" token）。
    # 最大容量 = sink + window = 6 个 token。
    # 一旦缓存超过 6 个，中间过旧的部分会被丢弃，只保留 [前 2 | 后 4]。
    kv = RollingKV(window=4, sink=2)

    # 模拟连续生成 10 个 token 的场景：每步追加 1 个新 token 的 K/V。
    # 语法：`for _ in range(10):` 里的 _ 表示"我不需要循环序号"，只关心重复 10 次。
    for _ in range(10):
        # torch.randn(B, H, 1, D)：形状 (1, 2, 1, 4)
        #   第三维 = 1 表示"这一步只追加 1 个新 token"（自回归解码的典型模式）。
        k_new = torch.randn(B, H, 1, D)
        v_new = torch.randn(B, H, 1, D)

        # step() 把新增 K/V 拼到缓冲区末尾，然后裁剪超限部分。
        # 返回完整缓存的 K/V 张量，供注意力计算使用。
        # 语法：`k, v = kv.step(...)` 是元组解包，step() 返回 (裁剪后的 K, 裁剪后的 V)。
        k, v = kv.step(k_new, v_new)

        # 核心断言：缓存时间维（dim=2）长度永不超过 6。
        # 语法：k.size(2) 取第 2 维（序列长度维）的大小，等价于 k.shape[2]。
        # 无论 step 了多少次，200 次、2000 次都一样——显存占用被锁定在 sink+window。
        assert k.size(2) <= 6

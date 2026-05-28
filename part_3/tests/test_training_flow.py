# ==========================================
# 测试：训练阶段 —— 完整前向 + 反向传播
# ==========================================
# 这是"训练视角"的集成测试，覆盖 RMSNorm + RoPE + SwiGLU + KV Cache
# 在训练模式下的协同工作。用户可以在这里设断点，单步跟踪：
#   - embedding 如何生成
#   - 每层 TransformerBlock 内部的 Pre-Norm → 注意力 → 残差 → FFN 流程
#   - loss 如何从 logits 和 targets 算出
#   - backward 时梯度如何流过各个组件

import torch
from model_modern import GPTModern


def test_full_model_training_forward_backward():
    """完整训练流程：前向传播 + 损失计算 + 反向传播。

    用极小模型 + 随机数据构造一个"模拟训练步"，
    验证所有组件在训练模式下协同正确。
    """
    # 极小模型：2 层，4 头，64 维嵌入，确保 CPU 上秒完成
    model = GPTModern(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        use_rmsnorm=True,   # ← 用 RMSNorm 替代 LayerNorm
        use_swiglu=True,    # ← 用 SwiGLU 替代 GELU FFN
        rope=True,          # ← 用 RoPE 替代学习型位置嵌入
        max_pos=128,    # RoPE 预计算的最大序列长度
        sliding_window=4,   # ← 开启滑动窗口注意力，每个 token 只关注最近 4 个位置
    )
    model.train()  # 设为训练模式（开启 Dropout 的逻辑，但因 dropout=0 实际无影响）

    # 构造假数据：batch=2, seq_len=8, 随机 token ID 在 [0, 255]
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))

    # ─── 断点建议：forward 入口 ───
    # 在这里设断点，进入 forward() 可以跟踪：
    #   tok_emb → blocks[n].forward()（内部：RMSNorm → QKV投影 → RoPE旋转 → 缩放点积注意力 → SwiGLU FFN）
    #   → ln_f → head → cross_entropy loss
    # kv_cache_list=None → 训练模式不使用 KV Cache，is_causal=True
    logits, loss, caches = model(idx, targets=targets, kv_cache_list=None, start_pos=0)

    # ─── 验证前向结果 ───
    # logits 形状：(B=2, T=8, vocab_size=256)，每个位置对每个词表 token 的得分
    assert logits.shape == (B, T, 256), f"Expected logits shape (2,8,256), got {logits.shape}"
    # 训练时每个层仍返回 KVCache 对象（只是没有历史而已），可用于调试
    assert len(caches) == 2, f"Should have 2 caches (one per layer), got {len(caches)}"
    # loss 是标量张量
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.ndim == 0, f"Loss should be a scalar, got ndim={loss.ndim}"
    assert loss.item() > 0, f"Cross-entropy loss should be > 0, got {loss.item()}"

    # ─── 断点建议：backward 入口 ───
    # 在这里设断点，进入 loss.backward() 可以跟踪梯度如何通过：
    #   head 投影 → ln_f → 各层 Block → tok_emb →
    #   在每个 Block 内部：FFN 权重 → 注意力 proj → QKV 投影权重
    loss.backward()

    # ─── 验证反向传播：各组件参数梯度正常 ───
    # 遍历所有命名参数，确认每个需要梯度的参数都拿到了梯度
    params_without_grad = []
    params_with_nan = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                params_without_grad.append(name)
            elif torch.isnan(param.grad).any():
                params_with_nan.append(name)

    assert len(params_without_grad) == 0, \
        f"These parameters have no gradient: {params_without_grad}"
    assert len(params_with_nan) == 0, \
        f"These parameters have NaN gradient: {params_with_nan}"

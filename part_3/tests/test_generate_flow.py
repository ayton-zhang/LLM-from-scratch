# ==========================================
# 测试：生成阶段 —— KV Cache 加速推理
# ==========================================
# 这是"生成视角"的集成测试，覆盖 RMSNorm + RoPE + KV Cache
# 在自回归解码中的协同工作。用户可以在这里设断点，单步跟踪：
#   - prefill 阶段：首次 forward，缓存全部 prompt 的 K/V
#   - decode 阶段：每步只算新 token，从缓存读历史 K/V
#   - KV Cache 的拼接与更新过程
#   - generate() 与 generate_nocache() 的内部差异

import torch
from model_modern import GPTModern


def _make_tiny_model():
    """创建一致的极小模型，供多个测试复用。"""
    return GPTModern(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        use_rmsnorm=True,   # ← RMSNorm 替代 LayerNorm
        use_swiglu=True,    # ← SwiGLU 替代 GELU
        rope=True,          # ← RoPE 位置编码
        max_pos=128,
    )


def test_generate_cache_vs_nocache_consistent():
    """KV Cache 版 generate() 与无缓存版 generate_nocache() 输出应完全一致。

    这是验证 KV Cache 实现正确性的"黄金测试"：
    如果缓存拼接/裁剪/RoPE start_pos 有任何错误，
    两种实现的输出对不齐（temperature=0 贪心解码时尤其敏感）。

    用户在 debug 时可以对比两边的：
    - RoPE 旋转角度（start_pos 计算是否正确）
    - K/V 拼接位置
    - 每步的 logits 值
    """
    model = _make_tiny_model()

    # 同一个 prompt：5 个 token
    prompt = torch.tensor([[10, 20, 30, 40, 50]])

    # temperature=0 确保贪心解码，每一步选择确定性的最高概率 token，
    # 消除随机采样带来的差异，使缓存版和无缓存版的对比有意义。
    out_cache = model.generate(
        prompt, max_new_tokens=8, temperature=0.0, top_k=50
    )
    out_nocache = model.generate_nocache(
        prompt, max_new_tokens=8, temperature=0.0, top_k=50
    )

    # 两者输出的 token 序列应逐位置完全一致
    assert torch.equal(out_cache, out_nocache), (
        "generate() and generate_nocache() outputs must match.\n"
        f"Cache:    {out_cache[0].tolist()}\n"
        f"NoCache:  {out_nocache[0].tolist()}"
    )


def test_generate_prefill_and_decode_cache_state():
    """验证 KV Cache 在 prefill → decode 两阶段中的状态变化。

    这是理解 KV Cache 工作流程的关键 debug 入口：
    - prefill（第一步）：整个 prompt 喂入模型，所有层的缓存从 None 变为 KVCache 对象，
      缓存的序列长度 = prompt 长度
    - decode（第二步）：只喂入 1 个新 token，每层缓存长度 +1，
      且上一步的旧缓存值被保留（而非被覆盖）
    """
    model = _make_tiny_model()

    prompt = torch.tensor([[10, 20, 30, 40, 50]])  # 5 个 token
    T_prompt = prompt.size(1)

    model.eval()

    # ─── Prefill 阶段：首次 forward，填充缓存 ───
    # 断点建议：在这里进入 forward()，观察 kv_cache_list=[None, None]，
    # 每层从头计算 K/V 并存入新缓存
    with torch.no_grad():
        logits_prefill, _, kvs = model(prompt, kv_cache_list=None, start_pos=0)

    # 所有层缓存都已初始化
    for i, cache in enumerate(kvs):
        assert cache is not None, f"Layer {i} cache should be initialized after prefill"
        # 缓存长度 = prompt 长度（prefill 阶段喂入了全部 5 个 token）
        assert cache.T == T_prompt, \
            f"Layer {i} cache T should be {T_prompt} after prefill, got {cache.T}"

    # 保存 prefill 阶段缓存的 K 值，用于后续验证
    prefill_k_layer0 = kvs[0].k.clone()

    # ─── Decode 阶段：每步只喂入最后 1 个 token ───
    # 断点建议：在这里进入 forward()，观察：
    #   idx_cond 只有 1 个 token（而非整个 prompt）
    #   start_pos = 5（已缓存的 token 数）
    #   RoPE 根据 start_pos=5 设置该 token 的旋转角度
    #   K/V 拼接：torch.cat([旧缓存, 新 K/V], dim=2)
    last_token = prompt[:, -1:]  # 模拟 generate() 第二步喂入的 token
    with torch.no_grad():
        logits_decode, _, kvs = model(last_token, kv_cache_list=kvs, start_pos=T_prompt)

    # 每层缓存长度 +1（新增 1 个 decode token）
    for i, cache in enumerate(kvs):
        assert cache.T == T_prompt + 1, \
            f"Layer {i} cache T should be {T_prompt + 1} after decode, got {cache.T}"

    # prefill 阶段的旧缓存值没被覆盖（前 5 个 token 的 K 值应与之前完全一致）
    assert torch.equal(kvs[0].k[:, :, :T_prompt, :], prefill_k_layer0), \
        "Prefill K values should be preserved after decode step"

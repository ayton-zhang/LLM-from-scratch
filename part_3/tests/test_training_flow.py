# ==========================================
# 测试：训练阶段 —— 完整前向 + 反向传播
# ==========================================
# 这是"训练视角"的集成测试，覆盖 RMSNorm + RoPE + SwiGLU + KV Cache
# 在训练模式下的协同工作。用户可以在这里设断点，单步跟踪：
#   - embedding 如何生成
#   - 每层 TransformerBlock 内部的 Pre-Norm → 注意力 → 残差 → FFN 流程
#   - loss 如何从 logits 和 targets 算出
#   - backward 时梯度如何流过各个组件
#
# 与 Part 2 的 GPT 训练测试相比，本文件测试了 Part 3 引入的现代组件：
#   - RMSNorm 替代 LayerNorm（归一化更快、无均值偏移）
#   - SwiGLU FFN 替代 GELU FFN（更强的非线性激活）
#   - RoPE 替代学习型位置嵌入（位置信息编码进 Q/K 的旋转角度中）
#   - 可选滑动窗口注意力（sliding_window），长序列显存优化
# 同时验证这些组件在训练模式下梯度能正常传播，不会出现 NaN 或断流。

import torch
from model_modern import GPTModern


def test_full_model_training_forward_backward():
    """完整训练流程：前向传播 + 损失计算 + 反向传播。

    用极小模型 + 随机数据构造一个"模拟训练步"，
    验证所有组件在训练模式下协同正确。

    为什么训练时不使用 KV Cache？
    - 训练是一次性处理整个序列（batch 内所有 token 同时参与计算），
      不需要"先算前 3 个 token，再算第 4 个"这种逐步缓存。
    - KV Cache 是为推理时的自回归解码设计的优化手段，
      训练时 `kv_cache_list=None`，is_causal=True 直接对整个序列做因果掩码。
    - 训练时每层虽然也会返回 KVCache 对象，但只是"初次填充"的缓存，
      不会被后续步骤复用。
    """
    # ==========================================
    # 模型构造：极小模型，CPU 上秒级完成
    # ==========================================
    # 用 2 层、4 头、64 维的极小配置，确保单次前向 + 反向在 CPU 上也秒完成，
    # 同时模型结构（Block 层数、多头注意力）仍能覆盖真实场景的代码路径。
    model = GPTModern(
        vocab_size=256,    # 词表大小：字节级 tokenizer 共 256 种可能的字节值
        block_size=64,     # 最大上下文长度：训练时序列最长 64 个 token
        n_layer=2,         # Transformer Block 层数：2 层足够验证多层堆叠的梯度流
        n_head=4,          # 注意力头数：4 头保证多头注意力的代码路径被覆盖
        n_embd=64,         # 隐藏层维度：64 维足够小但能正常跑通整个流程
        dropout=0.0,       # 丢弃率=0：关闭 Dropout，确保每次前向结果完全确定性
        use_rmsnorm=True,  # ← 用 RMSNorm 替代 LayerNorm（现代 Transformer 标配）
        use_swiglu=True,   # ← 用 SwiGLU 替代 GELU FFN（LLaMA/GPT-4 采用的激活方案）
        rope=True,         # ← 用 RoPE 替代学习型位置嵌入（更好的序列长度外推性）
        max_pos=128,       # RoPE 预计算的最大序列长度（远大于实际用的 8，验证外推设计）
        sliding_window=4,  # ← 开启滑动窗口注意力，每个 token 只关注最近 4 个位置
    )
    # 语法：model.train() 是 nn.Module 的方法，将自身及所有子模块的 training 属性设为 True。
    # 这会影响 Dropout（训练时随机丢弃，推理时关闭）和 BatchNorm（训练时用 batch 统计量，
    # 推理时用全局移动平均）。这里 dropout=0 所以实际无影响，但设 train() 是对训练语义的
    # 正确标注——如果后续把 dropout 调大，这步就会起效。
    model.train()

    # ==========================================
    # 构造训练数据：随机 token ID 模拟真实输入
    # ==========================================
    # 用 torch.randint 生成 [0, 255] 范围内的随机整数作为 token ID，
    # 模拟一个 batch=2, 序列长度=8 的训练样本。
    # B=2 验证 batch 维度正确处理（不会把 batch 维和序列维搞混）；
    # T=8 超过 sliding_window=4，能触发滑动窗口的局部注意力逻辑。
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))

    # ==========================================
    # 前向传播：从 token ID 到 loss 的完整流程
    # ==========================================
    # ─── 断点建议：forward 入口 ───
    # 在这里设断点，进入 forward() 可以跟踪完整的数据流动：
    #   tok_emb（查表出向量）
    #   → blocks[n].forward()（逐层，每层内部：RMSNorm → QKV投影 → RoPE旋转
    #     → 缩放点积注意力（带 sliding_window 局部窗口掩码）→ 残差连接
    #     → RMSNorm → SwiGLU FFN → 残差连接）
    #   → ln_f（Identity 透传，因为 use_rmsnorm=True）
    #   → head（线性投影到 vocab_size 维）
    #   → cross_entropy loss（对比 logits 和 targets）
    #
    # kv_cache_list=None：
    #   训练模式不使用 KV Cache。None 表示每层从头计算所有 token 的 K/V，
    #   注意力中 is_causal + sliding_window 的组合掩码直接作用于完整序列。
    # start_pos=0：
    #   RoPE 从位置 0 开始为这 8 个 token 编码绝对位置信息，
    #   确保第 i 个 token 的旋转角度与其在序列中的真实位置对齐。
    #
    # 语法：`logits, loss, caches = model(...)` 是多返回值元组解包。
    # forward 返回三个值：预测分数、损失、更新后的 KV Cache 列表。
    # 训练时损失是核心关注对象，logits 可用于检查输出分布，caches 用于调试。
    logits, loss, caches = model(idx, targets=targets, kv_cache_list=None, start_pos=0)

    # ─── 验证前向结果 ───
    # 语法：logits.shape 返回形状元组 (B, T, vocab_size)。
    # logits 的每一维含义：
    #   dim=0 (B=2)：batch 维，每个样本独立
    #   dim=1 (T=8)：时间/序列维，每个位置有独立的预测
    #   dim=2 (256)：词表维，每个值为对应 token 的原始得分（logit），越大越"被模型选中"
    assert logits.shape == (B, T, 256), f"Expected logits shape (2,8,256), got {logits.shape}"

    # 训练时每层仍返回 KVCache 对象（只是没有历史缓存而已），
    # 可用于调试：查看每层缓存了哪些 K/V 值。
    # 语法：len(caches) 等于 n_layer=2，因为 forward 中每层 append 了一个 cache。
    assert len(caches) == 2, f"Should have 2 caches (one per layer), got {len(caches)}"

    # loss 是标量张量（0 维），值来源于交叉熵：log_softmax(logits) 与 targets 的负对数似然。
    # 语法：loss.ndim 返回张量的维度数，标量（如 3.14）的 ndim=0。
    # loss.item() > 0：交叉熵恒为正（除非模型完美预测，概率 1.0 → 熵 0），
    # 随机初始化下 loss 约为 -ln(1/256) ≈ 5.54，远大于 0。
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.ndim == 0, f"Loss should be a scalar, got ndim={loss.ndim}"
    assert loss.item() > 0, f"Cross-entropy loss should be > 0, got {loss.item()}"

    # ==========================================
    # 反向传播：梯度从 loss 回流到所有参数
    # ==========================================
    # ─── 断点建议：backward 入口 ───
    # 在这里设断点，进入 loss.backward() 可以跟踪梯度如何通过：
    #   head 投影（线性层权重梯度）→ ln_f（透传，梯度直接回流）
    #   → 各层 Block（残差分支 + FFN 分支的梯度叠加）
    #     → 在每个 Block 内部：SwiGLU 门控梯度 → 注意力 proj 梯度
    #       → QKV 投影梯度 → RMSNorm 缩放参数梯度
    #   → tok_emb（词嵌入权重梯度，只有输入 token ID 对应行有非零梯度）
    #
    # 反向传播的过程：
    #   1. PyTorch 的 autograd 引擎从 loss（标量）出发，沿计算图逆向遍历
    #   2. 每个参与前向计算的张量操作（如 matmul、softmax、view）都有对应的反向函数
    #   3. 链式法则自动累积，最终每个 requires_grad=True 的参数得到 .grad 属性
    loss.backward()

    # ─── 验证反向传播：各组件参数梯度正常 ───
    # 遍历所有命名参数，逐项检查两类常见的梯度异常：
    #   1. 梯度为 None：该参数虽然 requires_grad=True，但没有参与计算图，
    #      可能是代码中某条路径把它跳过了（如 use_swiglu=False 时 SwiGLU 的权重）。
    #   2. 梯度含 NaN：某处计算出现了除零或 inf 传播到梯度中，
    #      常见原因如 attention 中 scale=0、或者 SwiGLU 激活值爆炸。
    #
    # 语法：model.named_parameters() 返回一个生成器，每次 yield (参数名, 参数张量) 的元组。
    # 与 .parameters() 的区别是多了参数名，便于定位"哪个参数出了问题"。
    # 参数名如 "tok_emb.weight"、"blocks.0.attn.qkv_proj.weight" 直接对应模型结构路径。
    params_without_grad = []
    params_with_nan = []
    for name, param in model.named_parameters():
        # requires_grad 是参数的属性，标识这个参数是否需要优化器更新。
        # embedding 权重、线性层权重默认 requires_grad=True；
        # 不需要训练的固定参数（如位置编码表）可设为 requires_grad=False 来跳过检查。
        if param.requires_grad:
            if param.grad is None:
                params_without_grad.append(name)
            # 语法：torch.isnan(param.grad) 逐元素检查是否 NaN，返回布尔张量，
            # .any() 把所有元素做逻辑或，只要有任意一个 NaN 就返回 True。
            elif torch.isnan(param.grad).any():
                params_with_nan.append(name)

    # 断言所有需要梯度的参数都拿到了有效的数值梯度。
    # 如果这里失败，通常意味着：
    #   params_without_grad 非空 → 某个模块的参数没有被 loss 的计算图包含（代码路径遗漏）
    #   params_with_nan 非空 → 梯度爆炸或除零问题（如 attention_sink 裁剪逻辑有 bug）
    assert len(params_without_grad) == 0, \
        f"These parameters have no gradient: {params_without_grad}"
    assert len(params_with_nan) == 0, \
        f"These parameters have NaN gradient: {params_with_nan}"

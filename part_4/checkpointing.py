from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))
import time
import torch
import shutil
import torch.nn as nn

# 默认 checkpoint 文件名。
# 约定 model_last.pt 永远指向“最近一次保存”的完整训练状态，
# 训练中断后默认从它恢复即可。
DEF_NAME = "model_last.pt"

# ----------------------------- TB-only helpers (safe no-ops otherwise) ----------------------------- #


# ==========================================
# TensorBoard 后端判断：只对 TBLogger 启用高级日志
# ==========================================
def _is_tb(logger) -> bool:
    # TBLogger 内部有 self.w = SummaryWriter(...)。
    # NoopLogger / WBLogger 没有可用的 TensorBoard writer，因此 getattr 返回 None。
    # 语法：getattr(obj, "w", None) 表示“取 obj.w；如果没有这个属性，就返回 None”。
    return getattr(logger, "w", None) is not None


# checkpointing._log_hparams_tb
def _log_hparams_tb(logger, args, total_steps):
    # 只在 TensorBoard 可用时记录超参数；其他 logger 直接跳过。
    # 这样训练脚本可以无脑调用，不需要自己判断日志后端。
    if not _is_tb(logger): return
    try:
        # 把 argparse 里的关键训练配置整理成一个普通 dict。
        # 这些值会显示在 TensorBoard 的 HParams 面板，方便比较多次实验：
        # 例如 lr=3e-4 / batch_size=32 / n_layer=2 哪个组合 loss 更低。
        h = dict(
            vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, lr=args.lr,
            warmup_steps=args.warmup_steps, batch_size=args.batch_size, grad_accum=args.grad_accum_steps,
            mixed_precision=args.mixed_precision, steps=args.steps, epochs=args.epochs,
        )
        # metrics_once 是配套的一次性指标。
        # 这里记录总训练步数，让 TensorBoard 知道这次 run 的训练规模。
        logger.hparams(h, {"meta/total_steps": float(total_steps)})
    except Exception:
        # 日志失败不应该中断训练；训练结果比可视化更重要。
        pass


# ==========================================
# 计算图日志：把模型结构写入 TensorBoard
# ==========================================
def _maybe_log_graph_tb(logger, model, xb, yb):
    # 不是所有 logger 都实现 graph()；没有这个方法就直接跳过。
    if not hasattr(logger, "graph"): 
        return
    try:
        # TensorBoard add_graph 需要模型 forward 返回 Tensor。
        # 但本项目的模型通常返回 (logits, loss, caches) 这样的 tuple，
        # 所以用一个轻量 wrapper 把输出整理成“第一个 Tensor”。
        class _TensorOnly(nn.Module):
            def __init__(self, m): 
                # 语法：super().__init__() 调用父类 nn.Module 的初始化逻辑，
                # 这样 _TensorOnly 才能像普通 PyTorch Module 一样被 trace。
                super().__init__(); self.m = m.eval()

            def forward(self, x, y=None):
                # y=None 时支持只传输入 x；有 y 时支持训练式调用 model(x, y)。
                out = self.m(x, y) if y is not None else self.m(x)
                if isinstance(out, (list, tuple)):
                    # 如果模型返回多个对象，优先找第一个 Tensor 作为 graph 输出。
                    # KV cache 这类列表/对象不适合直接交给 add_graph。
                    for o in out:
                        if torch.is_tensor(o):
                            return o
                    return out[0]
                return out

        # wrapper 放到和示例输入相同的设备上，避免 CPU/GPU 设备不一致。
        wrapped = _TensorOnly(model).to(xb.device)
        # 语法：(xb, yb) 是一个二元 tuple，表示 graph trace 时传入两个参数。
        logger.graph(wrapped, (xb, yb))
    except Exception:
        # 计算图 trace 对动态图/控制流很敏感，失败时不影响训练。
        pass


# ==========================================
# 模型统计日志：参数范数、梯度范数、可选直方图
# ==========================================
def _log_model_stats(logger, model, step: int, do_hists: bool = False):
    if not _is_tb(logger): return
    try:
        # 只统计 requires_grad=True 的可训练参数。
        # 冻结参数不会被优化器更新，混进来会让“训练状态”指标失真。
        params = [p for p in model.parameters() if p.requires_grad]

        # 每个参数张量先算自己的 L2 范数，再把所有参数范数堆起来算全局 L2。
        # 这相当于观察整模型权重规模是否异常变大。
        total_param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]), 2).item()

        # 梯度可能为 None：例如刚 zero_grad(set_to_none=True) 后，或者某些参数本轮未参与计算。
        grads = [p.grad for p in params if p.grad is not None]
        total_grad_norm = float('nan')
        if grads:
            # 全局梯度范数是训练健康度的重要指标：
            # 突然飙升常见于梯度爆炸；长期接近 0 可能说明梯度消失或学习率太低。
            total_grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()

        logger.log(step=step, **{
            "train/param_global_l2": total_param_norm,
            "train/grad_global_l2": total_grad_norm,
        })

        if do_hists:
            # 直方图更细：能看到参数/梯度分布是否偏移、塌缩、出现极端值。
            # 但写入开销更大，所以默认关闭。
            for name, p in model.named_parameters():
                logger.hist(f"params/{name}", p, step)
                if p.grad is not None:
                    logger.hist(f"grads/{name}", p.grad, step)
    except Exception:
        pass


# ==========================================
# 注意力 Q/K/V 日志：轻量观察各层投影分布
# ==========================================
def _maybe_log_attention(logger, model, xb, step: int, every: int = 100):
    """
    Logs Q/K/V histograms for each Transformer block using the current minibatch xb.
    No model edits. No hooks. Runs a light no-grad recomputation of the pre-attn path.
    - Takes first batch and first head only to keep logs tiny.
    - Uses pre-RoPE values (simpler & stable for histograms).
    """
    if not _is_tb(logger) or step == 0 or (step % every):
        return
    try:
        import torch
        # @torch.no_grad 的上下文版：这里是额外日志计算，不参与训练，
        # 所以不需要构建计算图，能省显存也避免污染梯度。
        #
        # autocast(enabled=False)：强制用 FP32 做这段日志统计。
        # 日志要看真实数值分布，没必要为了速度进入 AMP 的半精度路径。
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Recreate inputs seen by blocks
            # xb 是 token id，形状 (B, T)。
            # tok_emb(xb) 把每个 token id 查表成向量：形状 (B, T) → (B, T, C)。
            x = model.tok_emb(xb)           # (B,T,C)
            # drop 是训练时 embedding 后的 dropout；这里复用同一路径，让统计更接近真实 block 输入。
            x = model.drop(x)

            # 语法：B, T, _ = x.shape 是元组解包。
            # _ 表示最后一维 C 这里暂时不需要。
            B, T, _ = x.shape
            for li, blk in enumerate(getattr(model, "blocks", [])):
                # enumerate 同时给出层号 li 和 block 对象 blk。
                # getattr(model, "blocks", []) 表示如果模型没有 blocks 属性，就用空列表避免报错。
                #
                # Pre-LN Transformer 中，注意力前会先做 ln1。
                # h 是第 li 层注意力真正看到的归一化隐状态，形状仍为 (B, T, C)。
                h = blk.ln1(x)              # pre-attn normalized hidden

                attn = blk.attn
                # Project to Q/K/V exactly like the module (pre-RoPE for simplicity)
                # Q 投影后先 view 成多头形状，再 transpose 把 head 维提前：
                #   wq(h):      (B, T, C) → (B, T, n_head*d_head)
                #   view:       (B, T, n_head, d_head)
                #   transpose:  (B, n_head, T, d_head)
                q = attn.wq(h).view(B, T, attn.n_head,   attn.d_head).transpose(1, 2)      # (B,H,T,D)
                # K/V 可能使用 GQA/MQA，因此头数是 n_kv_head，不一定等于 n_head。
                # 形状：wq 类似，但 head 维变成 Hk。
                k = attn.wk(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)     # (B,Hk,T,D)
                v = attn.wv(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)     # (B,Hk,T,D)

                # Take a tiny slice to keep logs light
                # 只取第 1 个 batch、第 1 个 head，避免把完整 Q/K/V 都写进日志导致 event 文件暴涨。
                # q[:1, :1] 形状 (1, 1, T, D)，view(-1) 拉平成 (T*D,) 方便画直方图。
                # contiguous() 确保 transpose 后的内存布局连续，view 才能安全重塑。
                q1 = q[:1, :1].contiguous().view(-1).float().cpu()
                k1 = k[:1, :1].contiguous().view(-1).float().cpu()
                v1 = v[:1, :1].contiguous().view(-1).float().cpu()

                # Drop non-finite (defensive)
                # torch.isfinite 会筛掉 NaN / inf。
                # 如果模型数值爆炸，日志里至少不会因为非法值导致写入失败。
                q1 = q1[torch.isfinite(q1)]
                k1 = k1[torch.isfinite(k1)]
                v1 = v1[torch.isfinite(v1)]

                # 记录 Q/K/V 的分布直方图。
                # 如果某个张量被过滤到空，就跳过，避免 TensorBoard 对空数据报错。
                if q1.numel() > 0: logger.hist(f"qkv/block{li}/q_hist", q1, step)
                if k1.numel() > 0: logger.hist(f"qkv/block{li}/k_hist", k1, step)
                if v1.numel() > 0: logger.hist(f"qkv/block{li}/v_hist", v1, step)

                # Optional small scalars (norms) that show up on Time Series
                # square().mean().sqrt() 是均方根 RMS，用一个标量概括当前 Q/K/V 的典型幅度。
                if q1.numel(): logger.log(step=step, **{f"qkv/block{li}/q_l2_mean": float(q1.square().mean().sqrt())})
                if k1.numel(): logger.log(step=step, **{f"qkv/block{li}/k_l2_mean": float(k1.square().mean().sqrt())})
                if v1.numel(): logger.log(step=step, **{f"qkv/block{li}/v_l2_mean": float(v1.square().mean().sqrt())})

                # Advance x to next block with a CHEAP approximation to avoid doubling full compute:
                # use the model's own FFN path only; skip re-running attention (we're only logging pre-attn stats).
                # 这里不是完整复现 block.forward，而是便宜地推进到下一层附近：
                #   x + ffn(ln2(x))
                # 目的只是让下一层日志输入不要完全停留在第 0 层，减少额外计算成本。
                x = x + blk.ffn(blk.ln2(x))

    except Exception as e:
        print(f"[qkv] logging failed: {e}")


# ==========================================
# 运行时日志：吞吐量、耗时、GPU 显存
# ==========================================
def _log_runtime(logger, step: int, it_t0: float, xb, device):
    try:
        # 当前 step 从开始到现在耗时多少秒。
        dt = time.time() - it_t0
        # xb 形状通常是 (B, T)，numel() 就是本 micro-batch 的 token 数 B*T。
        toks = int(xb.numel())
        # tokens/s 是训练吞吐量，越高说明硬件利用越充分。
        # max(dt, 1e-6) 防止极端情况下除以 0。
        toks_per_s = toks / max(dt, 1e-6)
        # CUDA 可用时记录已分配显存，单位从 bytes 转成 MiB。
        # CPU 训练时记 0，保持日志字段一致。
        mem = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
        logger.log(step=step, **{
            "sys/throughput_tokens_per_s": toks_per_s,
            "sys/step_time_s": dt,
            "sys/gpu_mem_alloc_mb": mem
        })
    except Exception:
        pass


# ==========================================
# 文本采样日志：用当前模型生成一小段样例
# ==========================================
def _log_samples_tb(logger, model, tok, xb, device, step: int, max_new_tokens: int = 64):
    if not _is_tb(logger): return
    if tok is None: return
    try:
        # 切到 eval 模式：关闭 dropout，让采样结果更稳定。
        model.eval()
        with torch.no_grad():
            # xb[:1] 只取第一个样本作为 prompt，避免采样太慢。
            # generate 会在 prompt 后续写 max_new_tokens 个 token。
            out = model.generate(xb[:1].to(device), max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
        # 采样结束后切回 train 模式，恢复 dropout 等训练行为。
        model.train()
        # out[0].tolist() 把第一个样本的 token id 张量转成 Python list，
        # tokenizer 再把 token id 序列解码成人类可读文本。
        text = tok.decode(out[0].tolist())
        logger.text("samples/generation", text, step)
    except Exception:
        pass
# ---------------------------------------------------------------------- #


# ==========================================
# 模型配置提取：从现有模型反推可重建参数
# ==========================================
def _extract_config_from_model(model) -> dict:
    """
    Best-effort extraction of GPTModern-like config including GQA fields.
    """
    cfg = {}
    try:
        # 使用 getattr 做“宽容读取”：如果模型没有对应属性，就返回 None。
        # 这样这个函数不会强绑定某一个具体模型类。
        tok_emb = getattr(model, "tok_emb", None)
        blocks = getattr(model, "blocks", None)
        if tok_emb is None or not blocks:
            return cfg

        try:
            # SwiGLU 是可选模块；导入成功就用真实类判断，失败就造一个占位类。
            from swiglu import SwiGLU  # optional
        except Exception:
            class SwiGLU: pass

        # token embedding 的行数就是词表大小。
        cfg["vocab_size"] = int(tok_emb.num_embeddings)
        cfg["block_size"]  = int(getattr(model, "block_size", 0) or 0)
        # blocks 是 Transformer block 列表，长度就是层数。
        cfg["n_layer"]     = int(len(blocks))

        first_blk = blocks[0]
        attn = getattr(first_blk, "attn", None)
        if attn is None:
            return cfg

        # Heads & dims
        # n_embd = n_head * d_head，这是多头注意力里“所有头拼回去”的隐藏维度。
        cfg["n_head"]   = int(getattr(attn, "n_head"))
        d_head          = int(getattr(attn, "d_head"))
        cfg["n_embd"]   = int(cfg["n_head"] * d_head)
        # n_kv_head 用于 GQA/MQA。老模型可能没有这个属性，默认等于 n_head（标准 MHA）。
        cfg["n_kv_head"]= int(getattr(attn, "n_kv_head", cfg["n_head"]))  # default to MHA

        # Dropout (if present)
        # PyTorch Dropout 层的概率存在 .p；没有 dropout 层就记 0.0。
        drop = getattr(attn, "dropout", None)
        cfg["dropout"] = float(getattr(drop, "p", 0.0)) if drop is not None else 0.0

        # Norm/FFN style
        # 如果最终 norm 是 nn.Identity，说明模型可能把 RMSNorm 放在 block 内，
        # 或者使用了不需要额外 ln_f 的结构；这里把它记成 use_rmsnorm。
        cfg["use_rmsnorm"] = isinstance(getattr(model, "ln_f", None), nn.Identity)
        # 判断 FFN 是否使用 SwiGLU，用于恢复现代 Transformer 的前馈层类型。
        cfg["use_swiglu"]  = isinstance(getattr(first_blk, "ffn", None), SwiGLU)

        # Positional / attention tricks
        # 这些是现代注意力里的可选能力：RoPE、最大位置、滑动窗口、attention sink。
        # 只有模型真的有这些属性时才写入 config，避免老模型 checkpoint 多出无意义字段。
        for k in ("rope", "max_pos", "sliding_window", "attention_sink"):
            if hasattr(attn, k):
                val = getattr(attn, k)
                # bool 是 int 的子类；这里沿用原逻辑，把 bool 转成 0/1 写入。
                cfg[k] = int(val) if isinstance(val, bool) else val
    except Exception:
        # “best-effort” 的含义：提取失败就返回空配置，由上层决定如何处理。
        return {}
    return cfg


# ==========================================
# 架构校验：确认 checkpoint 与当前模型形状一致
# ==========================================
def _verify_model_matches(model, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, message)."""
    # expected 来自 checkpoint 中保存的 config，代表“当时训练这个权重的模型结构”。
    expected = {
        "block_size": cfg.get("block_size"),
        "n_layer":    cfg.get("n_layer"),
        "n_head":     cfg.get("n_head"),
        "n_embd":     cfg.get("n_embd"),
        "vocab_size": cfg.get("vocab_size"),
        "n_kv_head":  cfg.get("n_kv_head", cfg.get("n_head")),
    }
    # got 来自当前进程里已经构建好的 model，代表“你现在准备加载权重的模型结构”。
    got = {
        "block_size": int(getattr(model, "block_size", -1)),
        "n_layer":    int(len(model.blocks)),
        "vocab_size": int(model.tok_emb.num_embeddings),
    }
    first_blk = model.blocks[0]
    got.update({
        "n_head":     int(first_blk.attn.n_head),
        "n_embd":     int(first_blk.attn.n_head * first_blk.attn.d_head),
        "n_kv_head":  int(getattr(first_blk.attn, "n_kv_head", first_blk.attn.n_head)),
    })
    # 对关键结构字段逐项比较。
    # 例如 checkpoint 是 n_layer=4，而当前 model 是 n_layer=2，权重 shape 肯定对不上。
    diffs = [f"{k}: ckpt={expected[k]} vs model={got[k]}" for k in expected if expected[k] != got[k]]
    if diffs:
        return False, "Architecture mismatch:\n  " + "\n  ".join(diffs)
    return True, "ok"


# ==========================================
# 保存 checkpoint：模型、优化器、调度器、AMP、配置一起落盘
# ==========================================
def save_checkpoint(model, optimizer, scheduler, amp, step: int, out_dir: str,
                    tokenizer_dir: str | None = None, config: dict | None = None):
    # out 是 checkpoint 目录，例如 runs/part4。
    # mkdir(..., exist_ok=True) 保证目录不存在时创建，存在时继续复用。
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Prefer the model’s own config if available (e.g., a dict or dataclass with __dict__/asdict)
    # 优先保存 model.config，因为它通常是训练时最权威的配置来源。
    # 如果没有 model.config，就使用外部传入的 config；再不行才从模型结构反推。
    if hasattr(model, "config"):
        cfg_obj = model.config
        # 语法：三元表达式 A if 条件 else B。
        # 如果 config 本身是 dict，直接复制成普通 dict；
        # 否则尝试读取对象的 __dict__，例如 dataclass/简单配置类。
        cfg = dict(cfg_obj) if isinstance(cfg_obj, dict) else getattr(cfg_obj, "__dict__", None) or _extract_config_from_model(model)
    else:
        cfg = config if config is not None else _extract_config_from_model(model)

    # torch.save 会把 Python dict 序列化到 .pt 文件。
    # 这里保存的不只是模型权重，还包括“继续训练所需的全部状态”。
    torch.save({
        # model.state_dict()：所有可学习参数和 buffer，比如 embedding、attention 权重、norm 参数。
        "model": model.state_dict(),
        # optimizer.state_dict()：AdamW 的动量/方差统计等历史状态。
        # 不保存它，断点续训会像换了一个新优化器，前几步更新会不连续。
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        # scheduler 记录当前学习率曲线走到哪一步。
        # hasattr 是兼容性保护：有些调度器可能没有 state_dict 方法。
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        # AMP GradScaler 的缩放因子也要保存。
        # 否则恢复后 scaler 从默认值重新开始，可能短时间内不够稳定。
        "amp_scaler": amp.scaler.state_dict() if amp and getattr(amp, "scaler", None) else None,
        # step 是已经完成的 optimizer 更新次数，不是 micro-batch 次数。
        "step": int(step),
        "config": cfg,   # ← always write config
        # version 标记 checkpoint 格式，方便以后做兼容迁移。
        "version": "part4-v2",
    }, out / DEF_NAME)

    if tokenizer_dir is not None:
        # 保存 tokenizer 目录路径。模型权重必须配套同一个词表，
        # 否则 token id 的含义变了，embedding/head 的 vocab_size 也可能对不上。
        (out / "tokenizer_dir.txt").write_text(tokenizer_dir)


# ==========================================
# 加载 checkpoint：恢复权重并可选恢复训练状态
# ==========================================
def load_checkpoint(model, path: str, optimizer=None, scheduler=None, amp=None, strict: bool = True):
    # map_location="cpu" 让 checkpoint 先加载到 CPU。
    # 好处：GPU 训练保存的文件也能在 CPU 机器上读取，再由 model.to(device) 控制放到哪里。
    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt.get("config")
    if cfg:
        # 先校验架构，再加载权重。
        # 如果结构不一致，直接 load_state_dict 可能报一长串 shape mismatch，
        # 这里提前给出更清晰的差异说明。
        ok, msg = _verify_model_matches(model, cfg)
        if not ok:
            raise RuntimeError(msg + "\nRebuild the model with this config, or load with strict=False.")
    else:
        # Legacy checkpoint without config: strongly encourage a rebuild step elsewhere
        print("[compat] Warning: checkpoint has no config; cannot verify architecture.")

    # strict=True 时，checkpoint 里的参数名必须和当前模型完全对应。
    # 返回的 missing/unexpected 分别表示“当前模型缺少的权重”和“checkpoint 多出来的权重”。
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}")

    # 下面三段是“可选恢复”：
    # 只要调用方传入对应对象，并且 checkpoint 里有对应状态，就加载。
    # 这让同一个函数既能用于纯推理加载模型，也能用于训练断点续训。
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if amp is not None and ckpt.get("amp_scaler") is not None and getattr(amp, "scaler", None):
        amp.scaler.load_state_dict(ckpt["amp_scaler"])

    # 返回保存时的 optimizer step，训练循环据此从正确步数继续。
    return ckpt.get("step", 0)


# ----------------------------- checkpoint/save utils ----------------------------- #


# ==========================================
# checkpoint 文件路径：最新文件 + 按步数归档文件
# ==========================================
def checkpoint_paths(out_dir: Path, step: int):
    # f"model_step{step:07d}.pt" 会把 step 补成 7 位数字。
    # 例如 step=50 → model_step0000050.pt。
    # 这样文件名按字典序排序时，也会和训练步数顺序一致。
    return out_dir / f"model_step{step:07d}.pt", out_dir / "model_last.pt"


# ==========================================
# 保存全部状态并保留最近 K 个 step checkpoint
# ==========================================
def atomic_save_all(model, optim, sched, amp, step: int, out_dir: Path,
                    tok_dir: str | None, keep_last_k: int, config: dict):
    """Write model_last.pt (with config) + a rolling per-step copy."""
    # 第一步：写最新 checkpoint 到 model_last.pt。
    # save_checkpoint 内部会包含 model / optimizer / scheduler / AMP scaler / config / step。
    save_checkpoint(model, optim, sched, amp, step, str(out_dir), tok_dir, config=config)  # writes model_last.pt

    # 第二步：再复制一份带 step 编号的归档文件。
    # model_last.pt 方便“默认恢复最近一次”，model_stepxxxx.pt 方便回退到某个历史点。
    per_step, last = checkpoint_paths(out_dir, step)
    try:
        # copy2 会复制文件内容和部分元数据，例如修改时间。
        shutil.copy2(last, per_step)
    except Exception:
        # 归档副本失败时不影响 model_last.pt，至少最近 checkpoint 已经保存。
        pass

    # GC old per-step checkpoints
    try:
        # glob("model_step*.pt") 找出所有按步数归档的 checkpoint。
        # 因为文件名数字补零，sorted 后就是从旧到新的顺序。
        ckpts = sorted(out_dir.glob("model_step*.pt"))
        # 只保留最后 keep_last_k 个，前面的旧文件删除，防止磁盘被 checkpoint 填满。
        for old in ckpts[:-keep_last_k]:
            old.unlink(missing_ok=True)
    except Exception:
        pass

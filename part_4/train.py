# ==========================================
# 训练脚本：完整的 LLM 训练循环
# ==========================================
# Part 4 的核心文件——从零开始训练一个现代 GPT 模型。
#
# 整个训练流程（从头到尾的数据流）：
#   原始文本文件 → BPE Tokenizer 训练/加载 → 流式数据集 → batch 拼接
#   → 模型前向（AMP 混合精度）→ loss → 反向传播（梯度累积）
#   → optimizer.step() + scheduler.step() → 定期 checkpoint + 日志
#
# 本脚本的设计特点：
#   - 无 Trainer API：训练循环是显式的 while/for，适合学习底层机制
#   - 支持断点续训：从 checkpoint 恢复模型/优化器/调度器/AMP 状态
#   - 优雅退出：捕获 SIGINT/SIGTERM 信号，退出前自动保存
#   - 混合精度 + 梯度累积：小显存也能训练较大模型
#
# 用法示例：
#   python train.py --data ../part_2/tiny.txt --out runs/demo \
#     --bpe --vocab_size 8000 --batch_size 16 --block_size 128 \
#     --n_layer 2 --n_head 2 --n_embd 128 --steps 300 \
#     --mixed_precision --grad_accum_steps 2 --log tensorboard
from __future__ import annotations
import argparse, time, signal
from pathlib import Path
import sys

import torch
import torch.nn as nn

# ==========================================
# 路径设置：跨 part 目录导入 Part 3 的模型
# ==========================================
# so we can import Part 3 model
from pathlib import Path as _P
# parents[1] 向上两级：从 part_4/train.py → part_4/ → 项目根目录
# 再拼接 part_3/，加到 sys.path，使 `from model_modern import GPTModern` 能找到目标。
sys.path.append(str(_P(__file__).resolve().parents[1] / 'part_3'))
from model_modern import GPTModern

# Part 4 自己的模块
from tokenizer_bpe import BPETokenizer
from dataset_bpe import make_loader
from lr_scheduler import WarmupCosineLR
from amp_accum import AmpGrad
from checkpointing import (
    load_checkpoint,
    _log_hparams_tb,
    _maybe_log_graph_tb,
    _is_tb,
    _log_model_stats,
    _maybe_log_attention,
    _log_samples_tb,
    _log_runtime,
    atomic_save_all,
)
from logger import init_logger


# ==========================================
# 辅助函数：从命令行参数构造模型配置
# ==========================================
# 把 argparse 的扁平参数映射为 GPTModern 所需的嵌套 kwargs，
# 同时固定了一些 Part 4 的最佳实践默认值（开启 RMSNorm/SwiGLU/RoPE）。
def run_cfg_from_args(args, vocab_size: int) -> dict:
    return dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rmsnorm=True,       # Part 4 强制开启现代组件
        use_swiglu=True,
        rope=True,
        max_pos=4096,
        sliding_window=None,    # 训练时默认不开滑动窗口（简单全局注意力）
        attention_sink=0,
    )


# ==========================================
# main：训练主流程
# ==========================================
def main():
    # ==========================================
    # 命令行参数解析
    # ==========================================
    p = argparse.ArgumentParser()
    # 数据和输出路径
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='runs/part4')

    # tokenizer / model dims
    # --bpe：是否训练 BPE 分词器（推荐）。不加此 flag 则退回到字节级 tokenizer。
    p.add_argument('--bpe', action='store_true', help='train and use a BPE tokenizer (recommended)')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)

    # train
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps', type=int, default=300, help='max optimizer steps for this run')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=20)
    # --mixed_precision：启用 AMP（自动混合精度），前向用 FP16 加速，权重保持 FP32 精度。
    p.add_argument('--mixed_precision', action='store_true')
    # --grad_accum_steps：梯度累积步数。每 accum 步才做一次 optimizer.step()，
    # 等效于 batch_size * accum_steps 的大 batch 训练，但显存只需 1/accum。
    p.add_argument('--grad_accum_steps', type=int, default=4)

    # misc
    # choices=：限定参数只能从给定选项中选，传错会直接报错。
    p.add_argument('--log', choices=['wandb', 'tensorboard', 'none'], default='tensorboard')
    p.add_argument('--save_every', type=int, default=50, help='save checkpoint every N optimizer steps')
    # --keep_last_k：只保留最近 k 个 step checkpoint（加 model_last.pt 始终保留），节省磁盘。
    p.add_argument('--keep_last_k', type=int, default=2, help='keep last K step checkpoints (plus model_last.pt)')
    args = p.parse_args()

    # ─── 设备选择 ───
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 输出目录 + 断点续训检测
    # ==========================================
    # output dir and (possible) checkpoint
    # 语法：Path.mkdir(parents=True, exist_ok=True) 递归创建目录，已存在时不报错。
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_last.pt"
    have_ckpt = ckpt_path.exists()

    # ---- load checkpoint meta if present ----
    # 如果存在 checkpoint，加载模型权重和配置，实现断点续训。
    ckpt = None
    saved_tok_dir = None
    if have_ckpt:
        # torch.load(str(ckpt_path), map_location=device)：加载 checkpoint 到指定设备。
        # map_location 确保 GPU 训练出的 checkpoint 能在 CPU 上加载。
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "config" not in ckpt:
            raise RuntimeError(
                "Checkpoint is missing 'config'."
                "Please re-save a checkpoint that includes the model config."
            )
        # tokenizer_dir.txt 记录了分词器的保存路径，用于恢复分词器。
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # ==========================================
    # Tokenizer：BPE 训练 / 加载 / 字节级回退
    # ==========================================
    # ---- tokenizer ----
    tok = None
    tok_dir = None
    if have_ckpt:
        # 断点续训：必须加载原来的分词器（词表不一致会导致 embedding 维度错位）
        if not saved_tok_dir:
            raise RuntimeError(
                "Checkpoint was found but tokenizer_dir.txt is missing. "
                "Resume requires the original tokenizer."
            )
        tok = BPETokenizer(); tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[resume] Loaded tokenizer from {tok_dir} (vocab={vocab_size})")
    else:
        if args.bpe:
            # 全新训练：先在训练数据上训练 BPE 分词器。
            # BPE 从语料中学习"哪些字节组合最常出现"，合并为子词 token。
            # 相比字节级 tokenizer（vocab=256），BPE 能大幅压缩序列长度，
            # 同样的 block_size 能容纳更多语义信息。
            tok = BPETokenizer(vocab_size=args.vocab_size)
            tok.train(args.data)
            tok_dir = str(out_dir / 'tokenizer'); Path(tok_dir).mkdir(parents=True, exist_ok=True)
            tok.save(tok_dir)
            vocab_size = tok.vocab_size
            print(f"[init] Trained tokenizer to {tok_dir} (vocab={vocab_size})")
        else:
            # 字节级回退（Part 4 不推荐，BPE 效果更好）
            tok = None
            vocab_size = 256  # byte-level fallback (not recommended for Part 4)

    # ==========================================
    # 数据集：流式加载 + batch 拼接
    # ==========================================
    # make_loader 返回 DataLoader，内部用 BPE tokenizer 编码文本为 token ID，
    # 按 block_size 切片，shuffle=True 随机打乱防止模型记忆数据顺序。
    # ---- dataset ----
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size, shuffle=True)

    # ==========================================
    # 模型配置：新训练 vs 断点续训
    # ==========================================
    # ---- build model config ----
    if have_ckpt:
        cfg_build = ckpt["config"]
        # 安全检查：词表大小必须与 checkpoint 一致（否则 embedding 维度不匹配）
        if cfg_build.get("vocab_size") != vocab_size:
            raise RuntimeError(
                f"Tokenizer vocab ({vocab_size}) != checkpoint config vocab ({cfg_build.get('vocab_size')}). "
                "This deterministic script forbids vocab changes on resume."
            )
    else:
        cfg_build = run_cfg_from_args(args, vocab_size)

    # ==========================================
    # 初始化：模型、优化器、学习率调度器、AMP
    # ==========================================
    # ---- init model/opt/sched/amp ----
    # **cfg_build 语法：字典解包，等价于 GPTModern(vocab_size=..., block_size=..., ...)。
    model = GPTModern(**cfg_build).to(device)

    # AdamW 优化器：Adam 的改进版，decoupled weight decay。
    # betas=(0.9, 0.95)：Adam 的动量参数，LLaMA 的标准设置。
    # weight_decay=0.1：权重衰减，防止权重过大（正则化）。
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # 总步数 = min(用户指定的步数, 跑完所有 epoch 需要的步数)
    # len(train_loader) 返回每个 epoch 有多少个 batch。
    total_steps = min(args.steps, args.epochs * len(train_loader))

    # warmup：初始几轮学习率从 0 线性升到 base_lr，避免训练初期的不稳定。
    # 至少 1 步，最多 total_steps // 10（即 10% 的步数做 warmup）。
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    sched = WarmupCosineLR(optim, warmup_steps=warmup, total_steps=total_steps, base_lr=args.lr)

    # AmpGrad：封装 AMP（混合精度）+ 梯度累积的辅助类。
    # accum=args.grad_accum_steps：每 accum 步才真正更新参数。
    # amp=args.mixed_precision：是否开启 FP16 自动混合精度。
    amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

    # ==========================================
    # 断点续训：恢复训练状态
    # ==========================================
    # ---- strict resume ----
    step = 0
    if have_ckpt:
        # load_checkpoint：恢复模型权重、optimizer 动量、scheduler 状态、AMP scaler。
        # strict=True：如果 checkpoint 缺少某个组件就报错（而非静默跳过）。
        step = load_checkpoint(model, str(ckpt_path), optimizer=optim, scheduler=sched, amp=amp, strict=True)
        print(f"[resume] Loaded checkpoint at step {step}")

    # ==========================================
    # 日志初始化：TensorBoard / WandB / None
    # ==========================================
    # ---- logging ----
    logger = init_logger(args.log, out_dir=str(out_dir))
    # 记录超参数（args 和训练配置）到日志，方便后续对比实验。
    _log_hparams_tb(logger, args, total_steps)
    # 如果是 TensorBoard，尝试记录模型计算图（方便可视化模型结构）。
    if _is_tb(logger):
        try:
            # next(iter(...)) 取第一个 batch 作为示例输入。
            ex_x, ex_y = next(iter(train_loader))
            _maybe_log_graph_tb(logger, model, ex_x.to(device), ex_y.to(device))
        except Exception:
            pass

    # ==========================================
    # 信号处理：Ctrl+C 时优雅保存
    # ==========================================
    # ---- graceful save on SIGINT/SIGTERM ----
    # 用字典包装布尔值，因为闭包不能修改外部作用域的不可变类型（bool 是不可变的）。
    # 用 dict["flag"] 而非 bool 变量，确保回调函数内部能修改状态。
    save_requested = {"flag": False}
    def _on_term(sig, frame): save_requested["flag"] = True
    # signal.signal：注册信号处理器。SIGINT=Ctrl+C，SIGTERM=系统终止信号。
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)

    # ==========================================
    # 训练循环（整个脚本的核心）
    # ==========================================
    # ---- train loop ----
    # model.train()：设置训练模式，开启 Dropout。
    model.train()
    while step < args.steps:
        # 语法：`for xb, yb in train_loader:` 遍历 DataLoader 产生的 batch。
        # xb 形状 (B, T)，yb 是 xb 右移一位（下一个 token 预测任务）。
        # 每个 epoch 遍历一遍全部数据，外层 while 控制总步数。
        for xb, yb in train_loader:
            if step >= args.steps: break

            # ─── 信号检查：如果收到 Ctrl+C，保存后退出 ───
            if save_requested["flag"]:
                atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                print(f"[signal] Saved checkpoint at step {step} to {out_dir}. Exiting.")
                return

            # ─── 单步训练 ───
            it_t0 = time.time()
            # 把数据搬到 GPU（如果可用）
            xb, yb = xb.to(device), yb.to(device)

            # AMP 上下文管理器：
            # torch.cuda.amp.autocast(enabled=amp.amp)：自动将前向传播中的
            # 矩阵乘法等操作转为 FP16，速度翻倍、显存减半。
            # 只有 CUDA 设备才生效，CPU 上 enabled=False 时什么都不做。
            with torch.cuda.amp.autocast(enabled=amp.amp):
                # 前向传播：logits (B,T,vocab), loss (标量), caches (训练时不用)
                logits, loss, _ = model(xb, yb)

            # amp.backward(loss)：在 AMP 模式下用 GradScaler 缩放 loss 再反向传播，
            # 防止 FP16 梯度下溢（太小变成 0）。非 AMP 模式下等价于 loss.backward()。
            amp.backward(loss)

            # amp.should_step()：检查是否累积够了 grad_accum_steps 步。
            # 如果 accum=4，前 3 步只累积梯度不更新，第 4 步才真正更新。
            if amp.should_step():
                # amp.step()：取消梯度缩放 → 更新参数（optimizer.step）→ 更新 scaler。
                # amp.zero_grad()：清零梯度，为下一轮累积做准备。
                amp.step(); amp.zero_grad()
                # scheduler.step()：更新学习率（warmup 期间升到 base_lr，之后余弦衰减到 0）。
                lr = sched.step()
                step += 1

                # ─── 定期保存 checkpoint ───
                # periodic checkpoint
                if step % args.save_every == 0:
                    # atomic_save_all：先写到临时文件再原子重命名，防止保存过程中崩溃损坏文件。
                    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                    if _is_tb(logger):
                        logger.text("meta/checkpoint", f"Saved at step {step}", step)

                # ─── 定期记录日志 ───
                # logging
                if step % 50 == 0:
                    # 标量日志：loss、learning rate
                    logger.log(step=step, loss=float(loss.item()), lr=float(lr))
                    # 运行时日志：tokens/second、GPU 显存等
                    _log_runtime(logger, step, it_t0, xb, device)
                    # 模型统计：参数范数、梯度范数（检测梯度爆炸）
                    _log_model_stats(logger, model, step, do_hists=False)
                    # 注意力可视化（每 100 步一次，避免日志文件过大）
                    _maybe_log_attention(logger, model, xb, step, every=100)
                    # 文本采样：用当前模型生成一段文本，观察训练进展
                    _log_samples_tb(logger, model, tok, xb, device, step, max_new_tokens=64)

    # ==========================================
    # 训练结束：最终保存
    # ==========================================
    # ---- final save ----
    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
    print(f"Saved checkpoint to {out_dir}/model_last.pt")


# 语法：`if __name__ == '__main__':` 标准入口守卫。
if __name__ == '__main__':
    main()

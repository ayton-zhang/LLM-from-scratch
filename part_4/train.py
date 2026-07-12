# ==========================================
# 训练脚本：完整的 LLM 训练循环（手写版，无 Trainer 封装）
# ==========================================
# Part 4 的核心文件——从零开始训练一个现代 GPT 模型。
#
# ==========================================
# 整个训练流程的宏观数据流
# ==========================================
#
#  ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
#  │ 原始文本文件  │ →  │ BPE Tokenizer │ →  │ 流式 DataLoader│
#  │ (.txt)       │    │ (训练/加载)    │    │ (token ID 序列) │
#  └──────────────┘    └───────────────┘    └───────┬───────┘
#                                                    │
#            每个 batch: xb=(B,T), yb=xb 右移一位
#                                                    │
#  ┌──────────────┐    ┌───────────────┐    ┌───────▼───────┐
#  │ optimizer    │ ←  │ loss.backward │ ←  │ model.forward │
#  │ .step()      │    │ (梯度累积)     │    │ + loss 计算    │
#  └──────┬───────┘    └───────────────┘    └───────────────┘
#         │
#   scheduler.step()
#         │
#  ┌──────▼───────┐    ┌───────────────┐
#  │ checkpoint   │    │ logging       │
#  │ (定期保存)    │    │ (loss/lr/采样) │
#  └──────────────┘    └───────────────┘
#
# ==========================================
# Part 4 相比 Part 3 新增的训练基础设施
# ==========================================
#   BPE 分词器        ← 替代字节级 tokenizer，序列更短、语义更密集
#   流式数据集        ← 不把整个数据集加载到内存，边读边训
#   梯度累积           ← 小显存模拟大 batch：N 步累加梯度再一次性更新
#   AMP 混合精度       ← FP16 算矩阵乘法 + FP32 存权重，速度翻倍
#   学习率调度         ← Warmup（前 N 步从 0 升到 base_lr）+ Cosine 衰减到 0
#   检查点保存/恢复    ← 原子保存（先写临时文件再 rename），防止崩溃损坏
#   日志系统           ← TensorBoard / WandB，记录 loss/lr/grad_norm/采样文本
#
# ==========================================
# 设计哲学（为什么不用 HuggingFace Trainer？）
# ==========================================
#   - 显式训练循环（while/for）而非黑盒 Trainer，每一行都能设断点调试
#   - 无框架依赖（不依赖 accelerate / deepspeed / transformers）
#   - 适合学习：你清楚地看到"模型前向 → loss → backward → optimizer.step"
#     这四步的每一步，不会被框架的 callback/hook 机制绕晕
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
# Part 4 的模型定义在 Part 3（model_modern.py），需要通过 sys.path
# 让 Python 找到兄弟目录下的模块。这是一种"轻量级依赖管理"——
# 不用 pip install，直接修改 sys.path 指向目标目录。
# so we can import Part 3 model
from pathlib import Path as _P
# Path(__file__).resolve()：本文件的绝对路径
#      例：/home/yuteng/LLM-from-scratch/part_4/train.py
# .parents[1]：向上两级 → /home/yuteng/LLM-from-scratch/
# / 'part_3'：拼接为 /home/yuteng/LLM-from-scratch/part_3
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
# 辅助函数：命令行参数 → 模型配置字典
# ==========================================
# 为什么需要这个映射函数？
#   argparse 的参数是扁平的（args.n_layer, args.n_head, ...），
#   而 GPTModern.__init__ 需要嵌套的 kwargs。
#   run_cfg_from_args 做两件事：
#     1. 扁平参数映射为字典
#     2. 固定 Part 4 的最佳实践默认值（强制开启现代组件）
#
# 注意：训练时 sliding_window 和 attention_sink 都关掉了，
# 因为训练用的是全局因果注意力（T 较小，O(T²) 开销可接受），
# 滑动窗口主要用于推理时的 KV Cache 管理。
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
        sliding_window=None,    # 训练时不开滑动窗口
        attention_sink=0,
    )


# ==========================================
# main：训练主流程
# ==========================================
def main():
    # ==========================================
    # 命令行参数解析
    # ==========================================
    # argparse 的工作机制：
    #   p.add_argument('--foo', ...) 注册一个参数
    #   p.parse_args() 扫描 sys.argv，匹配参数名，提取值，返回命名空间对象
    # 此后通过 args.foo 访问值。
    p = argparse.ArgumentParser()
    # 数据和输出路径
    # required=True：必须提供，否则 argparse 直接报错并打印帮助信息。
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='runs/part4')

    # tokenizer / model dims
    # --bpe：action='store_true' → 传了为 True，没传为 False。布尔开关的标准写法。
    p.add_argument('--bpe', action='store_true', help='train and use a BPE tokenizer (recommended)')
    # vocab_size=32000：BPE 词表大小。32000 是 LLaMA-1 的标准设置，
    # 既能覆盖常见子词，又不会太大（vocab 太大会增加 embedding 参数量）。
    p.add_argument('--vocab_size', type=int, default=32000)
    # block_size=256：每个训练样本的最大 token 数。超过的被截断。
    # 256 是 Karpathy nanoGPT 的默认值，适合 CPU 训练和小数据集。
    p.add_argument('--block_size', type=int, default=256)
    # n_layer=6, n_head=8, n_embd=512：约 30M 参数的模型配置，能在消费级 GPU 上训练。
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    # dropout=0.0：现代大模型训练趋向于不用 dropout 或只用很小的值（0.05）。
    # 因为数据量足够大（或 epochs 少），过拟合不是主要问题。
    p.add_argument('--dropout', type=float, default=0.0)

    # train
    # batch_size=32：每个 step 的样本数。每个样本是 block_size 长度的 token 序列。
    p.add_argument('--batch_size', type=int, default=32)
    # epochs=1：遍历数据集几轮。通常语言模型只训 1 epoch（数据量足够大时），
    # 多 epoch 容易导致过拟合（模型背住训练数据而非学习语言规律）。
    p.add_argument('--epochs', type=int, default=1)
    # steps=300：最大 optimizer 步数。与 epochs 取 min，先达到谁就停止。
    p.add_argument('--steps', type=int, default=300, help='max optimizer steps for this run')
    # lr=3e-4：学习率。3e-4 是 LLaMA 论文的默认值，也是 AdamW + Transformer 的社区标准。
    # 比传统 SGD 的 1e-2 小很多，因为 Adam 会对梯度做自适应缩放。
    p.add_argument('--lr', type=float, default=3e-4)
    # warmup_steps=20：前 20 步学习率从 0 线性升到 base_lr。
    # 为什么需要 warmup？训练初期梯度方向和尺度都不稳定，直接用大学习率容易导致
    # 模型输出 NaN（梯度爆炸）。warmup 给优化器一个"缓冲期"建立动量统计量。
    p.add_argument('--warmup_steps', type=int, default=20)
    # --mixed_precision：启用 AMP。原理：
    #   前向传播的关键计算（matmul、conv）用 FP16（省一半显存，快一倍），
    #   但权重和梯度用 FP32 存储（保证精度）。
    #   GradScaler 在 backward 前把 loss 乘以一个大数（如 2^16），
    #   防止 FP16 的小梯度在累加时下溢为 0，optimizer.step 前再除以这个数恢复。
    p.add_argument('--mixed_precision', action='store_true')
    # --grad_accum_steps=4：每 4 个 micro-batch 才做一次 optimizer.step()。
    # 数学上等价于一次处理 batch_size*4 个样本（损失是各步 loss 之和 / 累积步数），
    # 但显存只需 1/4。代价是训练速度不变（因为总计算量相同），
    # 但允许在更小显存的 GPU 上训练更大 batch 的等效模型。
    p.add_argument('--grad_accum_steps', type=int, default=4)

    # misc
    # choices=[...]：限定参数值范围，传非法值 argparse 直接报错（不会静默忽略）。
    p.add_argument('--log', choices=['wandb', 'tensorboard', 'none'], default='tensorboard')
    # save_every=50：每 50 个 optimizer step 保存一次 checkpoint。
    # 太频繁 I/O 开销大；太稀疏崩溃损失大。50 是平衡点。
    p.add_argument('--save_every', type=int, default=50, help='save checkpoint every N optimizer steps')
    # keep_last_k=2：只保留最近 2 个 step checkpoint（如 step_100.pt, step_150.pt），
    # 加 model_last.pt 始终保留。避免磁盘被几百个 checkpoint 撑爆。
    p.add_argument('--keep_last_k', type=int, default=2, help='keep last K step checkpoints (plus model_last.pt)')
    args = p.parse_args()

    # ─── 设备选择 ───
    # torch.cuda.is_available()：检查 NVIDIA GPU 驱动 + CUDA 是否可用。
    # 有 GPU 就用 GPU（速度 10-50x），没有就退回到 CPU。
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 输出目录 + 断点续训检测
    # ==========================================
    # output dir and (possible) checkpoint
    # Path(args.out) 是面向对象的路径操作（如 out_dir / "foo.txt" 拼接路径）。
    # .mkdir(parents=True, exist_ok=True)：mkdir -p 的 Python 等价写法，
    #   递归创建所有父目录，已存在时不会报错。
    out_dir = Path(args.out) 
    out_dir.mkdir(parents=True, exist_ok=True)
    # 语法：/ 操作符在 Path 对象上被重载为路径拼接，比 os.path.join 更直观。
    ckpt_path = out_dir / "model_last.pt"
    have_ckpt = ckpt_path.exists()

    # ==========================================
    # 断点续训：加载 checkpoint 元数据
    # ==========================================
    # checkpoint 是一个普通 dict（torch.save 保存的就是 Python dict），包含：
    #   "config"   : 模型配置（vocab_size, n_layer, n_head, ...）
    #   "model"    : 模型权重 state_dict
    #   "optimizer": AdamW 的动量和二阶矩累积量
    #   "scheduler": 当前学习率、步数
    #   "amp"      : GradScaler 的缩放因子
    #   "step"     : 当前 optimizer step 编号
    # ---- load checkpoint meta if present ----
    # 如果存在 checkpoint，加载模型权重和配置，实现断点续训。
    ckpt = None
    saved_tok_dir = None
    if have_ckpt:
        # torch.load(str(ckpt_path), map_location=device)：
        #   map_location 确保 GPU 训练的 checkpoint 能在 CPU 上加载
        #   （自动把 CUDA tensor 转为 CPU tensor，或迁移到指定 CUDA 设备）。
        ckpt = torch.load(str(ckpt_path), map_location=device)
        # 防御性检查：老版本 checkpoint 可能没有 config 字段。
        if "config" not in ckpt:
            raise RuntimeError(
                "Checkpoint is missing 'config'."
                "Please re-save a checkpoint that includes the model config."
            )
        # tokenizer_dir.txt 记录了分词器的保存路径。
        # 断点续训时必须用同一分词器（词表变了 embedding 维度会错位）。
        # 语法：Path.with_name("新文件名") 只替换文件名部分，保留父目录不变。
        # 例如 ckpt_path = Path("ckpts/model.pt")
        #     → with_name("tokenizer_dir.txt") → Path("ckpts/tokenizer_dir.txt")
        # 效果等价于 ckpt_path.parent / "tokenizer_dir.txt"，但更简洁。
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # ==========================================
    # Tokenizer：BPE 训练 / 加载 / 字节级回退
    # ==========================================
    # BPE (Byte Pair Encoding) 的工作机制：
    #   1. 初始化：vocab 包含所有单个字节（0-255），共 256 个 token
    #   2. 遍历语料，统计"哪两个相邻 token 最常同时出现"
    #   3. 把出现最多的这对合并为新 token（如 "th" + "e" → "the"）
    #   4. 重复 step 2-3 直到词表达到 vocab_size
    #   结果：常见词如 "the" 占 1 个 token，罕见词如 "amazing" 可能拆成 "am"+"az"+"ing"
    #
    # 相比字节级 tokenizer (vocab=256)：
    #   - 序列更短（同样文本 token 数减少 2-4x）
    #   - 同样的 block_size 能装更多语义信息
    #   - 训练/推理更快（序列长度变短）
    #   - 代价：需要先训练 tokenizer（但只需一次，可复用）
    # ---- tokenizer ----
    tok = None
    tok_dir = None
    if have_ckpt:
        # 断点续训分支：必须加载原来的分词器。
        # 如果分词器丢失（只备份了 checkpoint 文件），模型无法恢复训练，
        # 因为 embedding 层的大小 = vocab_size * n_embd，词表变了维度就不对。
        if not saved_tok_dir:
            raise RuntimeError(
                "Checkpoint was found but tokenizer_dir.txt is missing. "
                "Resume requires the original tokenizer."
            )
        tok = BPETokenizer()
        tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[resume] Loaded tokenizer from {tok_dir} (vocab={vocab_size})")
    else:
        if args.bpe:
            # 全新训练分支：在训练数据上从头训练 BPE 分词器。
            # tok.train(args.data) 内部执行 BPE 合并算法（如上所述），
            # 产出 vocab_size 个 token 的合并规则。
            tok = BPETokenizer(vocab_size=args.vocab_size)
            tok.train(args.data)
            # 保存分词器到 out/tokenizer/，方便后续推理时加载。
            #
            # 语法：out_dir / 'tokenizer' 是 pathlib 的路径拼接操作符重载。
            # Path 对象重载了 / 运算符，使 a / 'b' 等价于 os.path.join(a, 'b')，
            # 但更简洁且可链式调用（如 out_dir / 'tokenizer' / 'vocab.json'）。
            #
            # str(out_dir / 'tokenizer')：Path → 字符串转换。
            # 为什么转成字符串？BPETokenizer.save(tok_dir) 的参数类型是 str，
            # 传给 Path(tok_dir).mkdir() 时又转回 Path——这是因为 mkdir 是 Path 的方法，
            # pathlib 风格更面向对象。
            #
            # .mkdir(parents=True, exist_ok=True)：
            #   - parents=True：递归创建父目录（等价于 shell 的 mkdir -p）。
            #     如果 out_dir 不存在，连 out_dir 一起创建，不用手动 mkdir(out_dir)。
            #   - exist_ok=True：目录已存在时不报错（不用先 os.path.exists 检查）。
            #     因为分词器可能已经训练过（缓存场景），不加 exist_ok 会抛 FileExistsError。
            tok_dir = str(out_dir / 'tokenizer')
            Path(tok_dir).mkdir(parents=True, exist_ok=True)
            tok.save(tok_dir)
            vocab_size = tok.vocab_size
            print(f"[init] Trained tokenizer to {tok_dir} (vocab={vocab_size})")
        else:
            # 字节级回退（Part 4 不推荐，保留只是为了兼容性）。
            # tok=None 传给 make_loader 时，后者会用原始字节编码。
            tok = None
            vocab_size = 256  # byte-level fallback (not recommended for Part 4)

    # ==========================================
    # 数据集：流式加载 + batch 拼接
    # ==========================================
    # make_loader 返回 torch.utils.data.DataLoader：
    #   1. 读取文本文件 → 2. tokenizer.encode() 转 token ID
    #   → 3. 按 block_size 切成长度固定的序列
    #   → 4. shuffle=True 随机排列 → 5. stack 成 (B, T) 的 batch
    #
    # yb 是 xb 右移一位（语言模型的"下一个 token 预测"任务）：
    #   xb = [tok0, tok1, tok2, ..., tok_{T-1}]
    #   yb = [tok1, tok2, tok3, ..., tok_T]
    #   模型看到 tok0 预测 tok1，看到 tok0..tok2 预测 tok3，以此类推
    # ---- dataset ----
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size, shuffle=True)

    # ==========================================
    # 模型配置：新训练 vs 断点续训
    # ==========================================
    # ---- build model config ----
    if have_ckpt:
        # 断点续训：用 checkpoint 中保存的 config 来重建相同结构的模型。
        # 不能改用新的 args（如改 n_layer），否则 state_dict 不匹配会报错。
        cfg_build = ckpt["config"]
        # 安全检查：虽然用了 checkpoint 的 config，但当前分词器的 vocab_size
        # 必须与 config 中的一致。不一致的唯一原因就是传了错误的分词器。
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
    # **cfg_build 语法：字典解包（dictionary unpacking）。
    # 等价于 GPTModern(vocab_size=256, block_size=256, ...)，
    # 但不需要一个一个写出所有键名。dict 的每个 key=value 自动变成函数的 keyword argument。
    # .to(device)：把模型的所有参数搬到 GPU（如果有），CPU 上不做任何事。
    model = GPTModern(**cfg_build).to(device)

    # AdamW 优化器详细机制：
    #   Adam = RMSProp + Momentum：对每个参数维护一阶动量 m（梯度均值）和二阶动量 v（梯度方差），
    #   利用 m 加速收敛方向，v 自适应调整每参数的学习率。
    #   AdamW = Adam + decoupled weight decay：传统 Adam 的 weight decay 与梯度耦合
    #   （L2 正则化混在自适应学习率里），AdamW 把 weight decay 从自适应更新中分离出来，
    #   直接对权重的原始值做衰减（w = w - lr * wd * w），效果更好（LLaMA/Mistral 标配）。
    #
    # betas=(0.9, 0.95)：动量衰减系数（LLaMA 论文的设置）。
    #   beta1=0.9：一阶动量的指数衰减率。m_t = 0.9 * m_{t-1} + 0.1 * g_t。
    #   beta2=0.95：二阶动量的指数衰减率（比标准 0.999 更激进，让自适应更快响应梯度变化）。
    # weight_decay=0.1：每步把权重的 10%*lr 衰减掉，防止权重只增不减。
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # 总步数 = min(用户指定的步数上限, 跑完所有 epoch 需要的步数)。
    # len(train_loader) = 数据集大小 // (batch_size * block_size)，每个 epoch 的 batch 数。
    total_steps = min(args.steps, args.epochs * len(train_loader))

    # warmup 计算：取 min(args.warmup_steps, total_steps // 10)。
    #   如果总步数很少（如只有 20 步），warmup 占比例太大会浪费训练时间，
    #   所以上限设 warmup_steps，下限至少 1 步（保证调度器不会在 step 0 报错）。
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    # WarmupCosineLR 的学习率曲线：
    #   步数 0..warmup：lr 从 0 线性增长到 base_lr （"热身"阶段）
    #   步数 warmup..total：lr 从 base_lr 余弦衰减到 0（"冷却"阶段）
    #   形状像一座山的轮廓： /‾‾‾‾‾‾\  （先上到顶，再慢慢下）
    # 为什么余弦衰减？比线性衰减更平滑——训练刚开始在"山顶"慢衰减（给模型时间探索），
    # 接近结束时快速衰减到 0（让模型收敛到最优解附近）。
    sched = WarmupCosineLR(optim, warmup_steps=warmup, total_steps=total_steps, base_lr=args.lr)

    # AmpGrad 内部维护了三个核心状态：
    #   1. step_counter：当前累积了多少步的梯度（累积够 accum 步才触发更新）
    #   2. scaler：torch.cuda.amp.GradScaler，管理 loss 缩放因子
    #   3. amp flag：是否启用 AMP（CPU 上应设为 False）
    amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

    # ==========================================
    # 断点续训：恢复完整训练状态
    # ==========================================
    # ---- strict resume ----
    step = 0
    if have_ckpt:
        # load_checkpoint 恢复的内容远不止模型权重：
        #   - model.load_state_dict(ckpt["model"])  → 模型参数
        #   - optimizer.load_state_dict(ckpt["optimizer"]) → Adam 的 m/v 累积量
        #       为什么 optimizer 也需要恢复？因为 Adam 的动量是训练历史的累积。
        #       如果不恢复，前 100 步的动量信息丢失，重启后前几步的更新方向会偏。
        #   - scheduler.load_state_dict(ckpt["scheduler"]) → 当前 lr 和步数
        #   - amp.scaler.load_state_dict(ckpt["amp"]) → GradScaler 的缩放因子
        # strict=True：如果 checkpoint 缺少以上任一组件就报错。
        #   设为 False 会静默跳过，可能导致训练从奇怪的状态继续（lr 不对、动量丢失等）。
        step = load_checkpoint(model, str(ckpt_path), optimizer=optim, scheduler=sched, amp=amp, strict=True)
        print(f"[resume] Loaded checkpoint at step {step}")

    # ==========================================
    # 日志初始化：TensorBoard / WandB / None
    # ==========================================
    # ---- logging ----
    # init_logger 根据 args.log 返回不同的 logger 对象：
    #   - "tensorboard" → SummaryWriter（写入 runs/part4/ 目录）
    #   - "wandb"       → wandb.init() 对象
    #   - "none"        → NoopLogger（所有 log 方法都是空操作）
    # 所有 logger 都实现了相同的 .log(step, loss, lr) 接口（鸭子类型），
    # 因此后面的日志代码不需要 if-else 判断 logger 类型。
    logger = init_logger(args.log, out_dir=str(out_dir))
    # 记录训练的超参数到日志，方便后续对比实验（如"lr=3e-4 vs lr=1e-3 哪个好"）。
    _log_hparams_tb(logger, args, total_steps)
    # 如果是 TensorBoard，记录模型的计算图（debug 时可以看到模型的层级结构）。
    # try/except：如果 graph 记录失败（如设备不支持、模型太大），静默跳过不中断训练。
    if _is_tb(logger):
        try:
            # next(iter(train_loader))：取 DataLoader 的第一个 batch。
            # iter() 创建迭代器，next() 取下一个元素。
            # 语法：`iter()` 和 `next()` 是 Python 迭代器协议的核心内置函数：
            #   iter(obj)  → 调用 obj.__iter__()，返回一个迭代器对象
            #   next(itr)  → 调用 itr.__next__()，返回下一个元素
            #   等价于 for 循环的第一步"取第一个 batch 就停下"，比 for + break 更直接。
            ex_x, ex_y = next(iter(train_loader))
            _maybe_log_graph_tb(logger, model, ex_x.to(device), ex_y.to(device))
        except Exception:
            pass

    # ==========================================
    # 信号处理：Ctrl+C 时优雅保存（不丢失训练进度）
    # ==========================================
    # ---- graceful save on SIGINT/SIGTERM ----
    # 问题：用户按 Ctrl+C 时 Python 默认直接退出，已经训练的几十分钟进度全丢。
    # 解决：注册信号处理器，收到 SIGINT/SIGTERM 时不直接退出，而是保存 checkpoint 后再退出。
    #
    # 为什么用 dict 包装布尔值而非直接用 bool 变量？
    #   信号处理器是回调函数。Python 闭包中，对不可变类型（bool/int/str）的赋值
    #   会创建新的局部变量，不修改外部作用域的值。而对可变类型（dict/list）的修改
    #   会影响外部。所以用 save_requested["flag"] = True 而非 save_requested = True。
    #   另一种方案是用 nonlocal 声明，但 dict 方案更通用（跨函数作用域）。
    save_requested = {"flag": False}
    def _on_term(sig, frame): save_requested["flag"] = True
    # SIGTERM：系统发送的终止信号（如 supervisor 重启服务、OOM killer 杀掉进程）
    # SIGINT：Ctrl+C 产生的终端中断信号
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)

    # ==========================================
    # ┌─────────────────────────────────────────┐
    # │         训练循环（整个脚本的核心）       │
    # └─────────────────────────────────────────┘
    # ==========================================
    # 训练循环的嵌套结构（while + for）：
    #
    #   while step < max_steps:       ← 外层：限制总步数（用户指定的 steps）
    #       for batch in dataloader:  ← 内层：遍历一个 epoch 的所有数据
    #           forward → backward → (accum) → step → log
    #
    # 为什么是 while step < max_steps 而不是 for epoch？
    #   因为用户可能想在 epoch 跑完前就停止（如只训练 300 步而非完整 1 epoch）。
    #   while 循环允许精确控制步数。
    #
    # 为什么 while 在外面而不是 for epoch in range(epochs)？
    #   for epoch 必须跑完整个数据集才能停止，不够灵活。
    #   如果数据很大（100GB），跑 1 epoch 需要几天，但用户只想快速验证 300 步。
    # ---- train loop ----
    # model.train()：开启训练模式。关键影响——
    #   1. Dropout 层生效（随机丢弃神经元）
    #   2. BatchNorm（如果用的话）用 batch 统计量而非全局平均值
    #   训练时一定要调 model.train()，否则 Dropout 不工作（dropout=0 时无影响）。
    model.train()

    while step < args.steps:
        # 语法：`for xb, yb in train_loader:` 在 for 循环中使用元组解包。
        # train_loader 每次 yield 一个 (xb, yb) 二元组，
        # Python 直接把元组的两个元素解包赋值给 xb 和 yb，
        # 比写 `for batch in loader: xb, yb = batch` 更简洁。
        # xb 是输入 token ID，yb 是目标 token ID（xb 右移一位）。
        # xb/yb 形状均为 (B, T)，dtype=torch.long。
        # 每轮 for 循环遍历完整个数据集（1 epoch），然后 while 循环判断是否继续。
        for xb, yb in train_loader:
            # 双重检查：for 循环内部也要检查步数上限（防止 DatLoader 没读完就超了）
            if step >= args.steps: break

            # ─── 信号检查：收到 Ctrl+C 时保存并退出 ───
            # 放在 for 循环内而非外层 while，因为 batch 处理可能很耗时，
            # 放在循环内能更快响应信号（每个 batch 开始前检查一次）。
            if save_requested["flag"]:
                atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                print(f"[signal] Saved checkpoint at step {step} to {out_dir}. Exiting.")
                return

            # ==========================================
            # 单步训练的 6 个核心步骤
            # ==========================================
            it_t0 = time.time()
            # .to(device)：如果数据在 CPU 上而模型在 GPU 上，把数据搬到 GPU。
            # 这是每步都必须做的操作（DataLoader 产出的数据默认在 CPU）。
            # 语法：`.to(device)` 把张量从 CPU 搬运到 GPU（如果已在目标设备上则无操作）。
            # xb 形状 (B, T)，dtype=long（token ID 必须是整数）。
            # yb 形状 (B, T)，是 xb 右移一位的结果——模型的任务是"看到 xb，预测 yb"。
            xb, yb = xb.to(device), yb.to(device)

            # ─── 第 1 步：前向传播（AMP 混精）───
            # torch.cuda.amp.autocast(enabled=amp.amp)：
            #   CUDA 上开启动器：自动将符合条件的操作（matmul, conv, linear）
            #   转为 FP16 执行。FP16 的 float 只有 16 位（vs FP32 的 32 位），
            #   表示范围小但精度足够前向传播。不是所有操作都转 FP16——
            #   softmax、normalization 等对精度敏感的操作仍用 FP32。
            #   非 CUDA 设备（CPU）上 enabled=False，整个上下文块什么都不做。
            with torch.cuda.amp.autocast(enabled=amp.amp):
                # model(xb, yb) 的内部调用链（张量形状变化追踪）：
                #   xb (B, T, dtype=long)
                #   → tok_emb: (B, T) → (B, T, C)         词嵌入查表
                #   → Block_0..Block_{N-1}: (B, T, C) → (B, T, C)  每层输出与输入形状相同
                #   → ln_f: (B, T, C) → (B, T, C)         最终归一化
                #   → head: (B, T, C) → (B, T, vocab_size) 投影到词表维度
                #   → cross_entropy: (B, T, vocab_size) vs yb (B, T) → 标量 loss
                #
                # 语法：`logits, loss, _ = model(...)` 是多返回值解包（Tuple Unpacking）。
                #   model.forward() 返回一个 (logits, loss, kvs) 三元组，
                #   Python 直接把三个元素分配给左边三个变量。
                #   下划线 `_` 是 Python 惯例——"我知道这里有返回值，但我不需要它"。
                #   训练时不需要 KV Cache（kvs 总为 None 的列表），用 _ 明确丢弃。
                #   logits: (B, T, vocab_size)  每个位置对每个 token 的预测得分
                #   loss: 标量，交叉熵 = -log(P(正确token|上文)) 的平均值
                #   kvs: [None, ..., None]  训练时 KV Cache 不激活
                logits, loss, _ = model(xb, yb)

            # ─── 第 2 步：反向传播（梯度累积）───
            # amp.backward(loss) 与 loss.backward() 的区别：
            #   - 非 AMP 模式：等价于 loss.backward()，计算梯度并累积到 param.grad
            #   - AMP 模式：先对 loss 乘以 scaler.get_scale()（如 2^16=65536），
            #     再调用 scaled_loss.backward()，这样 FP16 的小梯度被放大后
            #     不会在累加到 FP32 的 .grad 时下溢为 0。
            #     如果某次 backward 出现了 inf/NaN，scaler 会自动减半缩放因子
            #     并跳过本次 update（GradScaler 的自适应机制）。
            #
            # 梯度累积的原理：
            #   backward() 不会清零梯度，只会累加到 param.grad 上。
            #   累积 4 步后 .grad 里的值是 4 个 micro-batch 的梯度之和。
            #   然后 optimizer.step() 用这个"和梯度"更新一次参数。
            #   数学上：1 次大 batch 更新 ≈ 4 次小 batch 梯度累加后更新。
            #   误差来源：BatchNorm 的统计量（但 Transformer 不用 BN，所以几乎等价）。
            amp.backward(loss)

            # ─── 第 3 步：判断是否该更新参数 ───
            # amp.should_step() 检查 step_counter 是否达到 accum 步数。
            # 返回 True 时（如 accum=4，step_counter=3→0）触发更新。
            # 返回 False 时（step_counter 1,2,3）跳过，继续累积。
            if amp.should_step():
                # ─── 第 4 步：更新参数 ───
                # amp.step() 内部：
                #   1. 如果 AMP 开启：scaler.unscale_(optimizer) 取消梯度缩放
                #      （把 .grad 除以 scale 因子，恢复真实梯度值）
                #   2. optimizer.step()：AdamW 用恢复后的梯度更新所有参数
                #   3. scaler.update()：根据这步是否有 inf/NaN 调整缩放因子
                amp.step(); amp.zero_grad()

                # ─── 第 5 步：更新学习率 ───
                # sched.step() 根据当前步数和预设曲线（warmup→cosine）更新 optimizer 的 lr。
                # 每次 scheduler.step()，optimizer 每个参数组的 lr 都被重新计算。
                # 语法：sched.step() 返回当前步的学习率值（Python float），
                #   而非 PyTorch 张量。这是 WarmupCosineLR 特意设计的——
                #   内部用 float 算 lr 后直接返回，省去从张量中 .item() 的步骤。
                lr = sched.step()

                # step 只在真正更新参数后才 +1（梯度累积的中间步不计入 step）。
                step += 1

                # ─── 第 6 步：定期保存 checkpoint ───
                # periodic checkpoint
                if step % args.save_every == 0:
                    # atomic_save_all 的内部流程：
                    #   1. 把所有要保存的数据写到临时文件（.tmp 后缀）
                    #      {
                    #        "config": cfg_build,
                    #        "model": model.state_dict(),  ← 所有参数的权重值
                    #        "optimizer": optim.state_dict(),  ← Adam 的动量 m/v
                    #        "scheduler": sched.state_dict(),
                    #        "amp": amp.scaler.state_dict(),
                    #        "step": step,
                    #      }
                    #   2. 如果写成功，用 os.rename 把 .tmp 覆盖正式文件名
                    #      为什么用 rename 而不是直接写？rename 是原子操作——
                    #      要么旧文件还在，要么新文件完全写入。不会出现"写到一半
                    #      崩溃导致文件损坏"的情况。
                    #   3. 清理旧 checkpoint（只保留最近 keep_last_k 个 + model_last.pt）
                    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                    if _is_tb(logger):
                        logger.text("meta/checkpoint", f"Saved at step {step}", step)

                # ─── 定期记录日志（每 50 步一次）───
                # 为什么不是每步都打日志？
                #   - I/O 开销：TensorBoard 的 .log() 会写磁盘文件
                #   - 噪音：每步的 loss 波动很大（batch 不同），50 步平均更有参考价值
                #   - 显存：采样生成文字需要额外的前向传播，不能太频繁
                # logging
                if step % 50 == 0:
                    # 1. 基础标量：loss 和 learning rate
                    #    语法：loss.item() 把 0 维 PyTorch 张量（标量 tensor）转成 Python float。
                    #      PyTorch 张量即使只有 1 个值，也还连着计算图（占用显存），
                    #      .item() 把它从 GPU 搬出来、切断计算图，变成一个干净的 Python 数字。
                    #    float() 外层包装确保类型一致（.item() 本身返回 float，这里是防御性写法）。
                    logger.log(step=step, loss=float(loss.item()), lr=float(lr))
                    # 2. 运行时统计：tokens/second（吞吐量）、GPU 显存使用量
                    #    这是衡量"训练是否在有效利用硬件"的关键指标。
                    _log_runtime(logger, step, it_t0, xb, device)
                    # 3. 模型统计：权重范数、梯度范数
                    #    梯度范数突然飙升（如从 1.0 跳到 1000.0）→ 梯度爆炸，需要调低 lr
                    #    梯度范数持续接近 0 → 梯度消失，或学习率太低
                    #    do_hists=False：不记录参数/梯度的直方图分布（每 50 步画直方图太贵），
                    #    只记录标量统计量（L2 范数、最大值等），I/O 开销小得多。
                    _log_model_stats(logger, model, step, do_hists=False)
                    # 4. 注意力可视化（每 100 步一次，every=100）
                    #    记录各层注意力的平均模式——模型在关注哪些位置？
                    _maybe_log_attention(logger, model, xb, step, every=100)
                    # 5. 文本采样：用当前模型生成 64 个 token 的续写。
                    #    这是"最直观"的训练监控——看模型生成的文本质量是否在改善。
                    #    训练刚开始：乱码字节流；训练后期：逐渐出现单词、语法结构。
                    _log_samples_tb(logger, model, tok, xb, device, step, max_new_tokens=64)

    # ==========================================
    # 训练结束：最终保存
    # ==========================================
    # ---- final save ----
    # 训练完成（step 达到上限或数据遍历完毕），保存最终 checkpoint。
    # 不管用户有没有设 save_every，训练结束一定会保存 model_last.pt。
    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
    print(f"Saved checkpoint to {out_dir}/model_last.pt")


# 语法：`if __name__ == '__main__':` 标准入口守卫。
# 当 python train.py 时执行 main()，当 import train 时不执行。
if __name__ == '__main__':
    main()

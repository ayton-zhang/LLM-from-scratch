# ==========================================
# Part 6.5：SFT 训练脚本 — 最简单的单 GPU 微调循环
# ==========================================
# 本脚本实现了一个极简的 SFT（Supervised Fine-Tuning）训练流程。
# 核心思路：加载 Part 3/4 预训练好的 GPT 模型，在少量"指令→回答"对话数据上
# 继续训练几步，让模型从"纯文本补全器"变成"能听懂人类指令的助手"。
#
# 训练流程概览：
#   1. 加载 SFT 数据集（HuggingFace 切片或本地 fallback）
#   2. 用长度课程（curriculum）策略排序样本，由短到长逐步训练
#   3. 用 SFTCollator 将 (prompt, response) 对编码为 token 序列
#   4. 标准训练循环：前向 → 计算 loss → 反向传播 → 参数更新
#   5. 保存微调后的 checkpoint
#
# 与 Part 4 预训练的关键区别：
#   - 数据量：SFT 仅需几百到几千条高质量对话，预训练需要海量文本
#   - Loss 屏蔽：prompt 部分不参与 loss 计算（由 collator_sft.py 实现），
#     避免模型去"背诵"用户指令，只学习"在指令后生成合理回答"
#   - 学习率：SFT 用更小的学习率（3e-4），微调而非从头学习
#   - 训练步数：几百步即可，预训练需要数万步

from __future__ import annotations
import argparse, torch
import torch.nn as nn
from pathlib import Path

# 语法：torch.manual_seed(0) 固定随机数种子，确保每次运行结果可复现。
# 种子值 0 是惯例选择，任何固定整数都可以——关键是保持一致。
torch.manual_seed(0)

# ==========================================
# 导入 Part 3 的 GPTModern 模型
# ==========================================
# SFT 训练复用了 Part 3 中构建的现代 GPT 模型（带 RMSNorm + SwiGLU + RoPE），
# 不需要重新定义模型结构，只需加载预训练权重后继续训练即可。
# 这体现了模块化设计的价值：核心模型只定义一次，不同训练阶段（预训练/SFT）复用。
import sys
# 语法：from ... import ... as _P — 当需要避免命名冲突时给导入起别名。
# 上面已经 `from pathlib import Path`，这里用 `_P` 作为别名避免覆盖 Path。
from pathlib import Path as _P

# 语法：sys.path.append(str(...)) 将 part_3/ 目录加入 Python 模块搜索路径。
# __file__ 是当前文件的路径，.resolve() 转绝对路径，.parents[1] 取上两级目录（即 llm_from_scratch/），
# 然后拼接 'part_3'，使得 `from model_modern import GPTModern` 能正常工作。
# 这种跨目录引用的方式在小型项目中比 pip install -e . 更轻量，但不如包管理规范。
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402  # 忽略 flake8 的"导入不在文件顶部"警告

# ==========================================
# 导入 Part 6 各组件的本地模块
# ==========================================
# dataset_sft  → load_tiny_hf()：加载 SFT 对话数据集
# collator_sft  → SFTCollator：将 (prompt, response) 编码为模型输入，做 label masking
# curriculum    → LengthCurriculum：基于长度的课程学习排序器
from dataset_sft import load_tiny_hf
from collator_sft import SFTCollator
from curriculum import LengthCurriculum


# ==========================================
# main()：SFT 训练的主流程
# ==========================================
# 整个训练逻辑放在 main() 函数中而非模块顶层，这样：
#   1. 脚本可直接 `python train_sft.py` 运行
#   2. 也可被其他脚本 `from train_sft import main; main()` 调用（虽然目前未这样做）
# 这是 Python CLI 脚本的惯用模式。
def main():
    # ─── 命令行参数解析 ───
    # argparse 提供丰富的参数解析功能：类型检查（type=int/float）、
    # 布尔开关（action='store_true'）、默认值（default=）、帮助文本（help=）。
    p = argparse.ArgumentParser()

    # --data：数据集来源，默认从 HuggingFace 拉取。
    # 未来可扩展为本地 jsonl 路径，但当前 demo 主要使用 huggingface 模式。
    p.add_argument('--data', type=str, default='huggingface',
                   help='huggingface or path to local jsonl (unused in demo)')

    # --ckpt：预训练模型的 checkpoint 路径。
    # required=False 表示可选——没有 checkpoint 时从头初始化（随机权重），
    # 但在实际 SFT 流程中几乎总是需要加载预训练权重，否则模型没有语言能力基础。
    p.add_argument('--ckpt', type=str, required=False)

    # --out：训练输出目录，保存最终的 model_last.pt checkpoint
    p.add_argument('--out', type=str, default='runs/sft')

    # --steps：训练步数。SFT 数据量小，200 步已足够在少量样本上学到基本效果。
    # 步数过多可能导致过拟合（模型只会背训练样本的回答）。
    p.add_argument('--steps', type=int, default=200)

    # --batch_size：每个训练步的样本数。8 是单 GPU 的合理默认值。
    p.add_argument('--batch_size', type=int, default=8)

    # --block_size：最大序列长度（token 数）。256 对于问答任务通常够用。
    # 过长的 prompt+response 会被截断到此长度。
    p.add_argument('--block_size', type=int, default=256)

    # ─── 模型架构超参数 ───
    # 这些值需要与加载的预训练 checkpoint 保持一致！
    # 如果 checkpoint 是用 n_layer=2, n_embd=128 训练的，这里也必须匹配。
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)

    # --lr：学习率。3e-4（0.0003）是 SFT 的典型值，比预训练的常用值略低。
    # 预训练阶段模型学习通用知识需要更大步长，SFT 阶段只需微调方向，
    # 过大的学习率会冲散预训练学到的能力（灾难性遗忘）。
    p.add_argument('--lr', type=float, default=3e-4)

    # --cpu：强制使用 CPU 训练（即使检测到 GPU）。
    # action='store_true' 是布尔开关：不带 --cpu 时为 False，带了为 True。
    # 用于无 GPU 环境下的调试和验证，速度很慢但能确保代码逻辑正确。
    p.add_argument('--cpu', action='store_true')

    # --bpe_dir：BPE tokenizer 的路径。SFT 必须使用与预训练阶段完全相同的 tokenizer，
    # 否则 token ID 映射不同，预训练权重将完全无效（相当于随机初始化）。
    # 默认指向 Part 4 训练出的 tokenizer。
    p.add_argument('--bpe_dir', type=str,
                   default='../part_4/runs/part4-demo/tokenizer')

    # 语法：parse_args() 解析 sys.argv，返回包含所有参数值的 Namespace 对象。
    args = p.parse_args()

    # ─── 设备选择 ───
    # torch.cuda.is_available() 检测是否有 NVIDIA GPU + CUDA 驱动。
    # 语法：`A if 条件 else B` 是三元表达式，等价于：
    #   if torch.cuda.is_available() and not args.cpu:
    #       device = torch.device('cuda')
    #   else:
    #       device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ==========================================
    # 第一步：加载 SFT 数据集
    # ==========================================
    # load_tiny_hf() 从 HuggingFace 拉取一个对话数据集的小切片。
    # split='train[:24]' 表示只取训练集的前 24 条——极小数据集，仅用于 demo 验证流程。
    # sample_dataset=False 表示使用真实数据而非硬编码的 fallback 示例。
    # 返回的 items 是 SFTItem 列表，每个 item 包含 .prompt 和 .response 字段。
    items = load_tiny_hf(split='train[:24]', sample_dataset=False)

    # Print few samples
    # 打印前 3 条样本让用户直观了解数据集的内容和格式，方便调试。
    print(f"Loaded {len(items)} SFT items. Few samples:")
    for it in items[:3]:
        print(f"PROMPT: {it.prompt}\nRESPONSE: {it.response}\n{'-'*40}")

    # ==========================================
    # 第二步：构建长度课程采样器
    # ==========================================
    # 语法：列表推导式 [expr for var in iterable] 将 SFTItem 列表转为 (prompt, response) 元组列表。
    # LengthCurriculum 按 prompt+response 的总长度排序样本，从最短的开始训练。
    # 这类似于人类教学中的"由浅入深"：先让模型学习简短问答，再逐步增加长度。
    # 课程学习的优势：训练初期 loss 下降更平稳，避免模型一开始就被超长样本"吓到"。
    tuples = [(it.prompt, it.response) for it in items]
    # 语法：list(iterable) 将生成器/迭代器的所有元素收集到列表中。
    # LengthCurriculum 是一个可迭代对象（实现了 __iter__），list() 会遍历它收集所有批次。
    cur = list(LengthCurriculum(tuples))
    print(cur)

    # ==========================================
    # 第三步：构建数据整理器和模型
    # ==========================================
    # SFTCollator 负责将原始文本转为模型可吃的数字序列，并做 label masking。
    # block_size 控制最大序列长度，超长的 prompt+response 会被截断。
    # bpe_dir 指向训练好的 BPE tokenizer 目录（含 merges.txt 和 vocab.json）。
    col = SFTCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # 初始化 GPTModern 模型，使用 Part 3 的现代架构：
    #   use_rmsnorm=True  → 用 RMSNorm 替换 LayerNorm（更快、无需计算均值）
    #   use_swiglu=True   → 用 SwiGLU 替换 GELU FFN（更强的非线性表达能力）
    #   rope=True         → 用 RoPE 替换学习型位置嵌入（更好的序列长度外推性）
    # vocab_size 从 collator 的 tokenizer 属性读取，确保模型输出维度与 tokenizer 词汇表一致。
    # .to(device) 将模型所有参数和缓冲区移到指定设备（GPU 或 CPU）。
    model = GPTModern(vocab_size=col.vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)

    # ─── 加载预训练权重 ───
    # SFT 的关键一步：把 Part 4 预训练好的模型权重加载进来，
    # 这相当于给模型"装上预训练获得的大脑"，在此基础上做微调。
    if args.ckpt:
        print(f"Using model config from checkpoint {args.ckpt}")
        # torch.load() 用 pickle 反序列化保存的 checkpoint 字典。
        # map_location=device 在加载时就把张量移到目标设备，
        # 避免先在 CPU 加载再 .to(device) 的额外显存开销。
        ckpt = torch.load(args.ckpt, map_location=device)

        # ckpt.get('config', {}) 从 checkpoint 字典提取模型配置（如果不存在则返回空字典）。
        # 这里提取了配置但当前并未使用——它只是用于信息参考，
        # 确保用户知道预训练时用的什么超参数，手动校验命令行参数是否匹配。
        cfg = ckpt.get('config', {})

        # 语法：model.load_state_dict(ckpt['model']) 将 checkpoint 中保存的参数
        # 按名称精确覆盖到当前模型。如果参数名不匹配（如模型结构改变），会抛出异常。
        # 这是一种"严格模式"加载，避免静默的权重错位。
        model.load_state_dict(ckpt['model'])

    # ─── 优化器 ───
    # AdamW 是 Adam 的改进版，将权重衰减（weight decay）与自适应学习率解耦。
    # 参数说明：
    #   lr=3e-4         → 学习率，SFT 用小值防止冲散预训练知识
    #   betas=(0.9,0.95)→ Adam 的动量参数，控制梯度一阶/二阶矩的指数衰减速度
    #   weight_decay=0.1→ L2 正则化强度，0.1 是 LLaMA 系列的常用值，防止过拟合
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    # 语法：model.train() 将模型切换到训练模式。
    # 这主要影响 Dropout 和 BatchNorm 等层的行为：
    #   - Dropout 在训练时随机丢弃神经元，eval 时关闭
    #   - 当前模型中 Dropout=0.0（不用），但习惯上仍调用 train() 表示语义明确
    model.train()

    # ==========================================
    # 第四步：训练循环
    # ==========================================
    # 这是一个极简的单机训练循环，没有复杂的 dataloader、checkpoint 恢复、
    # 学习率调度等功能——力求代码最简洁，让读者聚焦核心流程。
    #
    # 循环逻辑：
    #   1. 从课程列表中取 batch_size 个 (prompt, response) 对
    #   2. collator 将它们编码为 token 序列并做 label masking
    #   3. 前向传播计算 loss
    #   4. 反向传播 + 参数更新
    #   5. 重复直到达到目标步数
    #
    # 当课程列表耗尽时，从头重新开始（简单循环而非打乱，对 demo 来说足够了）。
    step = 0   # 当前训练步数计数器
    i = 0      # 课程列表的取数据指针（类似于指针在数组中的位置）
    while step < args.steps:
        # 语法：列表切片 cur[start:end]，取从 i 开始的 batch_size 个元素。
        # 如果 i 超出列表范围（列表已耗尽），batch 为空列表 []，
        # Python 中空列表在布尔上下文中为 False。
        batch = cur[i:i+args.batch_size]
        if not batch:
            # 课程列表耗尽时重置指针，从头开始新一轮循环。
            # 注释掉的代码 `# cur = list(LengthCurriculum(tuples))` 是一个备选方案：
            # 重新打乱课程顺序（而非简单从头重复），但当前直接用 i=0 更简单。
            # restart curriculum
            # cur = list(LengthCurriculum(tuples));
            i = 0
            continue  # 语法：continue 跳过本次循环剩余代码，回到 while 开头

        # ─── 数据整理：文本 → 数字 ───
        # col.collate(batch) 将一批 (prompt, response) 对编码并整理为：
        #   xb：输入 token ID 张量，形状 (B, T+1)，T = block_size
        #   yb：目标 token ID 张量，形状 (B, T+1)，其中 prompt 位置被设为 -100（忽略）
        # 每个 batch 元素是 (prompt, response) 元组。
        xb, yb = col.collate(batch)
        # .to(device) 将数据从 CPU 移到 GPU，数据必须在与模型相同的设备上才能计算
        xb, yb = xb.to(device), yb.to(device)

        # ─── 前向传播 + Loss 计算 ───
        # model(xb, yb) 执行前向传播并同时计算交叉熵 loss。
        # 这比分开写 logits = model(xb); loss = F.cross_entropy(logits, yb) 更高效，
        # 因为模型内部可以复用 logits 的部分中间结果来计算 loss。
        # 返回值解包：
        #   logits：输出概率分布，形状 (B, T+1, vocab_size)
        #   loss：标量，平均交叉熵损失
        #   _：占位符，此处返回 None（Part 3 模型返回 kvs 用于推理，训练时不需要）
        logits, loss, _ = model(xb, yb)

        # ─── 参数更新三连：清零 → 反向传播 → 更新 ───
        # 1. opt.zero_grad(set_to_none=True)
        #    将优化器中累积的梯度清零。如果没有这一步，梯度会跨步累积（默认行为），
        #    导致参数更新方向错误。
        #    set_to_none=True 比 set_to_none=False 更高效：直接将 .grad 设为 None，
        #    PyTorch 会跳过 None 梯度的清零计算，节省时间和显存。
        opt.zero_grad(set_to_none=True)

        # 2. loss.backward()
        #    反向传播：从 loss 标量出发，通过计算图反向遍历，对每个 requires_grad=True
        #    的参数计算梯度（∂loss/∂参数），将结果累积到参数的 .grad 属性中。
        loss.backward()

        # 3. opt.step()
        #    优化器步进：AdamW 根据各参数的梯度（.grad）和学习率、动量等超参数，
        #    计算出参数更新量 Δθ，应用到模型参数上：θ_new = θ_old + Δθ。
        opt.step()

        # 步数计数器和数据指针都向前推进
        step += 1
        i += args.batch_size

        # 每 20 步打印一次 loss，让用户了解训练进度。
        # loss.item() 将形状为 () 的标量张量转为 Python float。
        # 如果 loss 在持续下降，说明训练正常；如果停滞或上升，可能需要调参。
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    # ==========================================
    # 第五步：保存 SFT 模型
    # ==========================================
    # 语法：Path.mkdir(parents=True, exist_ok=True)
    #   parents=True  → 自动创建所有不存在的父目录（类似 mkdir -p）
    #   exist_ok=True → 目录已存在时不报错（而非抛出 FileExistsError）
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 保存模型配置 + 权重到一个 checkpoint 文件。
    # 配置信息（cfg）记录了模型的架构参数和 tokenizer 信息，
    # 这样后续加载时无需重新指定超参数，可直接：
    #   ckpt = torch.load('model_last.pt')
    #   model = GPTModern(**ckpt['config'])  ← 自动恢复架构
    cfg = {
        "vocab_size": col.vocab_size,   # tokenizer 的词汇表大小（决定嵌入矩阵维度）
        "block_size": args.block_size,  # 最大序列长度
        "n_layer": args.n_layer,        # Transformer 层数
        "n_head": args.n_head,          # 注意力头数
        "n_embd": args.n_embd,          # 嵌入维度
        "dropout": 0.0,                 # Dropout 概率（SFT 阶段通常不用 dropout）
        "use_rmsnorm": True,            # 使用 RMSNorm（而非 LayerNorm）
        "use_swiglu": True,             # 使用 SwiGLU（而非 GELU）
        "rope": True,                   # 使用 RoPE（而非学习型位置嵌入）
        # ─── Tokenizer 信息（尽力而为的记录） ───
        # 根据词汇表大小推断 tokenizer 类型：256 对应 byte-level，其他对应 BPE。
        # 语法：`A if 条件 else B` 三元表达式，行内条件判断。
        "tokenizer_type": "byte" if col.vocab_size == 256 else "bpe",
        # tokenizer_dir 设为 None——如果需要完整保存 tokenizer 信息，
        # 应该将实际的 BPE 目录路径记录在此，方便后续推理时加载。
        "tokenizer_dir": None,
    }
    # torch.save() 用 Python 的 pickle 序列化任意 Python 对象。
    # 这里保存一个字典，包含两个键：
    #   'model'  → state_dict（OrderedDict，参数名 → 参数张量）
    #   'config' → 模型配置字典
    torch.save({'model': model.state_dict(), 'config': cfg},
               str(Path(args.out)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {args.out}/model_last.pt")


# ==========================================
# 脚本入口
# ==========================================
# 语法：`if __name__ == '__main__'` 是 Python 的惯用法，确保以下代码
# 仅在直接运行此文件时执行，被 import 时不执行。
# 这样其他脚本可以安全地 `from train_sft import main` 而不触发训练流程。
if __name__ == '__main__':
    main()

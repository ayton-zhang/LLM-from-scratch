# ==========================================
# 奖励模型 (Reward Model, RM) 单卡极小规模训练脚本
# 职责：加载人类偏好数据（Prompt + Chosen + Rejected），利用 PairCollator 组装成对 Token 张量，
#       通过 Bradley-Terry 或 Margin-Ranking 损失优化 Transformer 编码器，使其为偏好回复赋予更高标量得分。
# ==========================================

from __future__ import annotations
import argparse, torch
from pathlib import Path

from data_prefs import load_preferences
from collator_rm import PairCollator
from model_reward import RewardModel
from loss_reward import bradley_terry_loss, margin_ranking_loss


def main():
    # ─── 1. 命令行参数解析 ───
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/rm-demo')        # 权重与配置保存的输出目录
    p.add_argument('--steps', type=int, default=500)                 # 训练迭代总步数
    p.add_argument('--batch_size', type=int, default=8)              # 每批次处理的偏好样本对数量
    p.add_argument('--block_size', type=int, default=256)            # 上下文窗口的最大序列长度 (Prompt + Response)
    p.add_argument('--n_layer', type=int, default=4)                 # Transformer 编码器层数
    p.add_argument('--n_head', type=int, default=4)                  # 多头注意力机制的头数
    p.add_argument('--n_embd', type=int, default=256)                # 词嵌入与隐层维度
    p.add_argument('--lr', type=float, default=1e-4)                 # AdamW 优化器的学习率
    p.add_argument('--loss', choices=['bt','margin'], default='bt')  # 损失函数类型：'bt' (Bradley-Terry) 或 'margin' (Margin Ranking)
    p.add_argument('--cpu', action='store_true')                     # 强制使用 CPU 进行计算的标志位
    p.add_argument('--bpe_dir', type=str, default=None)              # 自定义 BPE 分词器目录（若为 None 则使用默认分词器）
    args = p.parse_args()

    # ─── 2. 计算设备准备 ───
    # 语法：优先检测 GPU (CUDA) 是否可用且未开 `--cpu` 标志，否则回退至 CPU 计算
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # data
    # ─── 3. 人类偏好数据集加载 ───
    # 演示目的：仅加载前 80 条偏好样本进行极小规模快速训练（Micro-training）
    items = load_preferences(split='train[:80]')
    # 语法：列表推导式解包结构化数据为 (Prompt, Chosen, Rejected) 三元组列表
    # 数据流动：PreferenceItem 对象列表 → 纯文本元组列表
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    # collator + model
    # ─── 4. 数据整理器、模型与优化器初始化 ───
    # PairCollator 负责将文本元组分词并填充对齐为固定长度的 Token 张量
    col = PairCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # 实例化奖励模型并将权重移至计算设备 (GPU/CPU)
    # 输入：词表大小、序列切片长度、网络深度/宽度参数
    model = RewardModel(vocab_size=col.vocab_size, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd).to(device)

    # 初始化 AdamW 优化器（适用于 Transformer 的权重衰减修正优化器）
    # 语法/原理：betas=(beta1, beta2) 分别控制梯度的：
    #   - beta1 (0.9)  : 一阶矩衰减率（动量 Momentum，平滑梯度方向，保留 90% 历史方向惯性）
    #   - beta2 (0.999): 二阶矩衰减率（方差 Variance，自适应调节各参数学习率，以 99.9% 比例平滑梯度平方幅度）
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # train (tiny)
    # ─── 5. 极小规模训练循环 ───
    step = 0; i = 0
    while step < args.steps:
        # 手动切片模拟 DataLoader 批次迭代
        batch = triples[i:i+args.batch_size]
        # 若已切片完当前数据集末尾，将索引重置为 0 重新开启一轮 (Epoch)
        if not batch:
            i = 0; continue

        # 数据整理 (Collate)：将批次文本转化成正/负样本的 Token ID 张量
        # pos 形状: (B, T) - 包含 Prompt + Chosen(胜出回复)
        # neg 形状: (B, T) - 包含 Prompt + Rejected(败选回复)
        pos, neg = col.collate(batch)
        # 将张量数据从内存/CPU 搬运到指定的计算设备 (GPU 显存)
        pos, neg = pos.to(device), neg.to(device)

        # 前向传播 (Forward Pass)：计算模型给出的标量奖励得分
        # r_pos 形状: (B,) - 正样本回复的质量得分
        # r_neg 形状: (B,) - 负样本回复的质量得分
        r_pos = model(pos)
        r_neg = model(neg)

        # 损失计算 (Loss Calculation)：
        if args.loss == 'bt':
            # Bradley-Terry 损失：-log(sigmoid(r_pos - r_neg))，鼓励 r_pos 远高于 r_neg
            loss = bradley_terry_loss(r_pos, r_neg)
        else:
            # 边距排序损失：max(0, margin - (r_pos - r_neg))，要求差距至少达到 margin(1.0)
            loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)

        # 优化器梯度清零：
        # 设计决策：`set_to_none=True` 将梯度设为 None 而非全零张量，能略微节省显存与提升速度
        opt.zero_grad(set_to_none=True)
        # 反向传播 (Backward Pass)：根据损失标量计算各层参数的梯度
        loss.backward()
        # 梯度更新：根据计算出的梯度调整模型参数权重
        opt.step()

        step += 1; i += args.batch_size

        # ─── 6. 日志打印与批次准确率评估 ───
        if step % 25 == 0:
            # 语法：(r_pos > r_neg) 返回布尔张量 (True/False)
            # .float() 转为 1.0/0.0，.mean() 计算当前 Batch 内正样本得分高于负样本的比例，.item() 转化为 Python float
            acc = (r_pos > r_neg).float().mean().item()
            print(f"step {step}: loss={loss.item():.4f} acc={acc:.2f}")

    # ─── 7. 模型权重与配置持久化 ───
    # 确保输出目录存在
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # 保存 PyTorch 检查点（包含模型权重 state_dict 和重建网络所需的 config 字典）
    torch.save({'model': model.state_dict(), 'config': {
        'vocab_size': col.vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved reward model to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()
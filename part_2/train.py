from __future__ import annotations  # 允许在类型提示中使用字符串形式的类型
import argparse, time  # argparse 用于命令行参数解析，time 用于计时
import torch  # PyTorch 深度学习库
from tokenizer import ByteTokenizer  # 字节级分词器
from dataset import ByteDataset  # 数据集加载器
from model_gpt import GPT  # GPT 模型类


def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    """评估模型在训练集和验证集上的平均损失。"""
    model.eval()  # 设置模型为评估模式（关闭 dropout 等）
    out = {}  # 存储结果的字典
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算
        for split in ['train', 'val']:  # 分别评估训练集和验证集
            losses = []  # 存储每次迭代的损失
            for _ in range(args.eval_iters):  # 评估 eval_iters 次
                xb, yb = ds.get_batch(split, args.batch_size, args.device)  # 获取一个 batch 的数据
                _, loss = model(xb, yb)  # 前向传播，返回预测和损失（忽略预测）
                losses.append(loss.item())  # 提取标量值并保存
            out[split] = sum(losses) / len(losses)  # 计算平均损失
    model.train()  # 恢复训练模式
    return out  # 返回包含 train 和 val 损失的字典


def main():
    """主函数：解析参数、初始化模型、训练模型。"""
    # 创建命令行参数解析器
    p = argparse.ArgumentParser()
    # === 数据和路径参数 ===
    p.add_argument('--data', type=str, required=True)  # 数据文件路径（必需）
    p.add_argument('--out_dir', type=str, default='runs/min-gpt')  # 输出目录，保存模型和日志
    # === 模型架构参数 ===
    p.add_argument('--block_size', type=int, default=256)  # 上下文窗口大小（输入序列长度）
    p.add_argument('--batch_size', type=int, default=32)  # 每个 batch 的样本数
    p.add_argument('--n_layer', type=int, default=4)  # Transformer 层数
    p.add_argument('--n_head', type=int, default=4)  # 多头注意力的头数
    p.add_argument('--n_embd', type=int, default=256)  # 嵌入维度（模型宽度）
    p.add_argument('--dropout', type=float, default=0.0)  # Dropout 比率
    # === 训练参数 ===
    p.add_argument('--steps', type=int, default=2000)  # 总训练步数
    p.add_argument('--lr', type=float, default=3e-4)  # 学习率
    p.add_argument('--weight_decay', type=float, default=0.1)  # AdamW 优化器的权重衰减
    p.add_argument('--grad_clip', type=float, default=1.0)  # 梯度裁剪阈值
    p.add_argument('--eval_interval', type=int, default=200)  # 每多少步评估一次
    p.add_argument('--eval_iters', type=int, default=50)  # 评估时的迭代次数
    # === 采样参数 ===
    p.add_argument('--sample_every', type=int, default=200)  # 每多少步采样一次
    p.add_argument('--sample_tokens', type=int, default=256)  # 每次采样生成的 token 数
    p.add_argument('--temperature', type=float, default=1.0)  # softmax 温度，控制采样的多样性（大=多样，小=确定）
    p.add_argument('--top_k', type=int, default=50)  # Top-k 采样：只考虑概率最高的 k 个 token
    p.add_argument('--top_p', type=float, default=None)  # Top-p 采样：核采样，概率累积到 p 就停止
    # === 计算设备和优化参数 ===
    p.add_argument('--cpu', action='store_true')  # 强制使用 CPU（不使用 GPU）
    p.add_argument('--compile', action='store_true')  # 使用 torch.compile 加速（PyTorch 2.0+）
    p.add_argument('--amp', action='store_true')  # 启用混合精度训练（自动混合精度，AMP）
    args = p.parse_args()  # 解析命令行参数

    # === 设置计算设备 ===
    # 如果有 GPU 且没有指定 --cpu，则使用 CUDA；否则使用 CPU
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # === 初始化分词器、数据集和模型 ===
    tok = ByteTokenizer()  # 创建字节级分词器
    ds = ByteDataset(args.data, block_size=args.block_size)  # 加载数据集，block_size 是上下文长度
    # 创建 GPT 模型并移到指定设备（GPU 或 CPU）
    model = GPT(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)

    # === 可选：模型编译优化（PyTorch 2.0+） ===
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)  # 使用 torch.compile 加速，需要 PyTorch 2.0+

    # === 优化器和梯度缩放器 ===
    # AdamW 优化器：带权重衰减的 Adam，betas 控制动量
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # 混合精度训练的梯度缩放器：防止 float16 下梯度溢出或下溢
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == 'cuda'))

    # === 训练前准备 ===
    best_val = float('inf')  # 记录最好的验证损失
    t0 = time.time()  # 记录开始时间戳
    model.train()  # 设置模型为训练模式
    # === 主训练循环 ===
    for step in range(1, args.steps + 1):  # 从 1 到 steps（包含）
        # 获取一个 batch 的训练数据
        xb, yb = ds.get_batch('train', args.batch_size, args.device)
        # === 混合精度前向传播 ===
        # 启用 AMP 时，计算自动转为 float16 以加速；否则保持 float32
        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.type == 'cuda')):
            _, loss = model(xb, yb)  # 前向传播，获得损失（忽略预测值）
        # === 反向传播和优化 ===
        opt.zero_grad(set_to_none=True)  # 清空上一步的梯度（set_to_none=True 比赋值 0 更高效）
        scaler.scale(loss).backward()  # 缩放损失后反向传播，计算梯度
        # === 梯度裁剪（防止梯度爆炸） ===
        if args.grad_clip > 0:  # 如果设置了梯度裁剪
            scaler.unscale_(opt)  # 恢复梯度的原始大小（取消缩放）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪梯度范数，保证 ≤ grad_clip
        # === 更新参数 ===
        scaler.step(opt)  # 优化器步进（自动处理缩放）
        scaler.update()  # 更新缩放器的缩放因子

        # === 定期日志输出（每 50 步） ===
        if step % 50 == 0:
            # 输出当前步数、损失和经过的时间
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time()-t0):.1f}s")
            t0 = time.time()  # 重置计时器

        # === 定期评估（每 eval_interval 步） ===
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)  # 计算训练集和验证集的平均损失
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
            # === 保存最优模型 ===
            if losses['val'] < best_val:  # 如果验证损失更小
                best_val = losses['val']  # 更新最好的验证损失
                ckpt_path = f"{args.out_dir}/model_best.pt"  # 构造检查点路径
                import os; os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录
                # 保存模型权重和配置信息到文件
                torch.save({'model': model.state_dict(), 'config': {
                    'vocab_size': tok.vocab_size,  # 词汇表大小
                    'block_size': args.block_size,  # 上下文长度
                    'n_layer': args.n_layer,  # Transformer 层数
                    'n_head': args.n_head,  # 多头注意力的头数
                    'n_embd': args.n_embd,  # 嵌入维度
                    'dropout': args.dropout,  # dropout 比率
                }}, ckpt_path)  # 保存到文件
                print(f"saved checkpoint: {ckpt_path}")

        # === 定期采样并生成文本 ===
        if args.sample_every > 0 and step % args.sample_every == 0:
            # 从训练集中随机选择一个起点（避免超出边界）
            start = torch.randint(low=0, high=len(ds.train) - args.block_size - 1, size=(1,)).item()
            # 提取 block_size 长的种子序列，作为初始上下文
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            # 模型生成文本：输入种子，生成 sample_tokens 个新 token
            out = model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            # 将 token 序列解码为文本
            txt = tok.decode(out[0].cpu())
            # 打印生成的文本（最后 block_size + sample_tokens 个字符，包括种子和生成部分）
            print("\n================ SAMPLE ================\n" + txt[-(args.block_size + args.sample_tokens):] + "\n=======================================\n")

    # === 最终保存 ===
    # 训练完成后，保存最终模型（仅包含权重，不含配置）
    import os; os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model': model.state_dict()}, f"{args.out_dir}/model_final.pt")


# === 程序入口点 ===
# 确保只有直接运行此脚本时才执行主函数（如果被导入则不执行）
if __name__ == '__main__':
    main()  # 运行主函数
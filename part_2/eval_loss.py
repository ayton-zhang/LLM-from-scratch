from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    # torch.compile 训练并保存后，参数键名可能变成 "_orig_mod.xxx"。
    # 为了兼容未 compile 的 GPT 结构，这里做键名规范化。
    if state_dict and all(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # 字典推导式：{新键: 原值 for 旧键, 原值 in 原字典.items()}
        # k[len('_orig_mod.'):] 是字符串切片，表示从前缀长度位置开始截取。
        # 例如 "_orig_mod.head.weight" 会变成 "head.weight"。
        return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    # 普通 checkpoint（没有该前缀）直接原样返回。
    return state_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = ByteDataset(args.data, block_size=args.block_size)
    # map_location=device：加载时把权重直接映射到目标设备（CPU 或 CUDA）。
    ckpt = torch.load(args.ckpt, map_location=device)
    # 统一处理参数键名，兼容 compile / 非 compile 的 checkpoint。
    state_dict = _normalize_state_dict_keys(ckpt['model'])
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    model.load_state_dict(state_dict)

    # eval() 会把模型切到评估模式（如 Dropout 关闭），用于稳定评估损失。
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()
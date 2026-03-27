from __future__ import annotations
import argparse, torch
from tokenizer import ByteTokenizer
from model_gpt import GPT


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    # torch.compile 训练后的模型在保存参数时，key 可能变成 "_orig_mod.xxx"。
    # 这里做一次“键名规范化”：如果发现全部 key 都带该前缀，就统一去掉前缀。
    if state_dict and all(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # dict 推导式语法：{新键: 原值 for 旧键, 原值 in 原字典.items()}
        # k[len('_orig_mod.'):] 是字符串切片：从前缀长度位置开始截取，得到去前缀后的键名。
        # 例如 "_orig_mod.tok_emb.weight" -> "tok_emb.weight"
        return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    # 如果不是 compile 产物（或混合异常情况），直接原样返回。
    return state_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    prompt_ids = tok.encode(args.prompt).unsqueeze(0).to(device)
    if prompt_ids.numel() == 0:
        # If no prompt provided, seed with newline byte (10)
        prompt_ids = torch.tensor([[10]], dtype=torch.long, device=device)


    # torch.load(..., map_location=device) 可在加载时把张量映射到目标设备（CPU/CUDA）。
    ckpt = torch.load(args.ckpt, map_location=device)
    # 统一处理 checkpoint 的参数键名，兼容 compile / 非 compile 两种保存格式。
    state_dict = _normalize_state_dict_keys(ckpt['model'])
    config = ckpt.get('config', None)

    if config is None:
        # fall back to defaults
        model = GPT(tok.vocab_size, block_size=256).to(device)
        model.load_state_dict(state_dict)
    else:
        model = GPT(**config).to(device)
        model.load_state_dict(state_dict)

    # 切到评估模式：关闭 Dropout 等训练期行为，保证采样稳定可复现。
    model.eval()

    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(tok.decode(out[0].cpu()))


if __name__ == '__main__':
    main()
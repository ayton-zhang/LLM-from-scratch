# ==========================================
# Demo：端到端文本生成演示
# ==========================================
# 本脚本用一个小型 GPTModern 模型生成一段文本，
# 展示 RMSNorm + RoPE + SwiGLU + KV Cache + 滑动窗口等
# 所有 Part 3 组件的端到端协同工作。
#
# 核心对比：同时运行带 KV Cache 的 generate() 和无缓存的 generate_nocache()，
# 验证两者输出一致（KV Cache 实现正确的"黄金测试"），
# 并对比两者耗时差异（Cache 版应明显更快，尤其在长序列时）。
#
# 用法示例（从 part_3/ 目录运行）：
#   python demo_generate.py --rmsnorm --rope --swiglu
#   python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200
#
# 参数含义：
#   --rmsnorm        : 启用 RMSNorm（否则用 LayerNorm）
#   --rope           : 启用 RoPE 位置编码
#   --swiglu         : 启用 SwiGLU FFN（否则用 GELU MLP）
#   --sliding_window : 滑动窗口大小，None = 全局注意力
#   --sink           : 注意力水槽大小（StreamingLLM）
#   --group_size     : KV 头数（GQA），默认 2（4 个 Q 头分 2 组，每组共享 1 对 KV）
#   --tokens         : 生成多少个新 token（默认 120）
#   --cpu            : 强制用 CPU（即使有 GPU）
import argparse, torch
from tokenizer import ByteTokenizer
from model_modern import GPTModern
import time

# 语法：`if __name__ == "__main__":` 标准入口守卫，只在直接运行脚本时执行。
if __name__ == "__main__":
    # ==========================================
    # 命令行参数解析
    # ==========================================
    p = argparse.ArgumentParser()
    # action='store_true'：传了该 flag 则 args.xxx=True，否则默认 False。
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--rope', action='store_true')
    p.add_argument('--swiglu', action='store_true')
    # type=int：参数值是整数类型。
    # default=None：不传 --sliding_window 时默认全局注意力。
    p.add_argument('--sliding_window', type=int, default=None)
    p.add_argument('--sink', type=int, default=0)
    p.add_argument('--group_size', type=int, default=2)
    p.add_argument('--tokens', type=int, default=120)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    # ─── 设备选择 ───
    # torch.cuda.is_available()：检查是否有可用的 NVIDIA GPU。
    # --cpu 强制使用 CPU（适合无 GPU 的环境或在笔记本上测试）。
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ==========================================
    # 模型构建
    # ==========================================
    # ByteTokenizer：字节级分词器，vocab_size=256（0-255 每个字节一个 token）。
    tok = ByteTokenizer()

    # 一个小型模型（2 层，4 头，128 维），参数约 0.5M，CPU 上也能快速运行。
    # n_kv_head=args.group_size：演示 GQA，默认 2 个 KV 头共享给 4 个 Q 头。
    model = GPTModern(
        vocab_size=tok.vocab_size,
        block_size=128,       # 最大上下文 128 token
        n_layer=2,            # 2 层 Transformer Block
        n_head=4,             # 4 个 Query 注意力头
        n_embd=128,           # 隐藏维度 128
        use_rmsnorm=args.rmsnorm,      # 是否用 RMSNorm
        use_swiglu=args.swiglu,        # 是否用 SwiGLU
        rope=args.rope,                # 是否用 RoPE
        max_pos=4096,                  # RoPE 最大支持 4096 位置
        sliding_window=args.sliding_window,  # 滑动窗口大小
        attention_sink=args.sink,      # 注意力水槽
        n_kv_head=args.group_size      # KV 头数（GQA）
    ).to(device)  # 把模型参数搬到 device（CPU 或 GPU）

    # ─── Prompt 构造 ───
    # 用一个空行（token ID=10，对应换行符 '\n'）作为 prompt。
    # 为什么不是空 prompt？不能传空张量给模型（T 至少为 1）。
    # 用换行符作为"无意义"的起始 token，让模型自由发挥。
    # empty prompt → newline
    prompt = torch.tensor([[10]], dtype=torch.long, device=device)

    # ==========================================
    # 生成阶段：带 KV Cache vs 无缓存
    # ==========================================
    # torch.no_grad() 上下文管理器：禁用梯度计算。
    # 推理时不需要梯度，关闭后能节省大量显存和计算时间。
    # 如果用 `with torch.no_grad():` 包装，里面的所有张量运算都不会构建计算图。
    with torch.no_grad():
        # ─── 带 KV Cache 的生成 ───
        # temperature=0.0：贪心解码（永远选概率最高的 token），确定性输出。
        # top_k=50：只从概率 top-50 的候选 token 中采样。
        # 打印耗时，可以看到 KV Cache 带来的加速效果。
        start = time.time()
        out = model.generate(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        # 语法：f"{time.time()-start:.2f}" 格式化字符串，:.2f 保留两位小数。
        print(f"Generated {args.tokens} tokens in {time.time()-start:.2f} sec")

        # ─── 无缓存的生成（对比基准）───
        # 相同 prompt、相同参数，输出应该完全一致。
        # 耗时通常比 cache 版长很多（每一步都重算整个序列的注意力）。
        start = time.time()
        out_nocache = model.generate_nocache(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"(nocache) Generated {args.tokens} tokens in {time.time()-start:.2f} sec")

    # ==========================================
    # 解码输出：把 token ID 转回可读文本
    # ==========================================
    # out[0]：取 batch=0（唯一一个样本），形状 (1+T_generated,)。
    # .cpu()：如果张量在 GPU 上，先搬到 CPU 才能给 tokenizer 解码。
    # tok.decode() 把 token ID 序列转回字节串，再按 UTF-8 解码为可读字符串。
    #
    # 同时打印两版输出，方便对比是否一致。
    # 如果一致，说明 KV Cache 实现正确；
    # 如果不一致，说明缓存拼接/裁剪/RoPE start_pos 有 bug。
    print(tok.decode(out[0].cpu()))
    print(tok.decode(out_nocache[0].cpu()))

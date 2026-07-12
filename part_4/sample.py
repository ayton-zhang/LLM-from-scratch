# ==========================================
# 模型采样脚本：从训练好的检查点加载模型进行文本生成
# ==========================================
# 这个脚本是 Part 4 的"推理入口"——它不参与训练，只负责：
#   1. 从 .pt 检查点文件加载训练好的模型权重
#   2. 加载对应的 BPE 分词器（如果存在）
#   3. 接收用户输入的 prompt，让模型续写文本
#
# 核心流程：
#   加载检查点 → 推断/读取配置 → 构建模型 → 加载权重 → 编码 prompt → 自回归生成 → 解码输出
#
# 与直接训练脚本的区别：本脚本需要"反向推断"模型配置——
# 因为旧检查点可能没有保存 config 字典，必须从权重的形状反推架构参数。

# 语法：`from __future__ import annotations` 延迟注解求值（PEP 563），
# 让类型注解变为字符串，减少 import 时的计算开销，同时支持前向引用。
from __future__ import annotations
import argparse, torch
from pathlib import Path

# ==========================================
# 加载 Part 3 的模型定义
# ==========================================
# part_4/ 下的脚本需要引用 part_3/ 中的 GPTModern 类。
# 由于它们不在同一目录，Python 默认找不到 part_3/，需要手动把它的路径加入 sys.path。
import sys
from pathlib import Path as _P  # 语法：`as _P` 是别名导入，避免与上面的 `Path` 冲突
# 语法：__file__ 是当前脚本的绝对路径；resolve() 解析符号链接；parents[1] 取上两级目录（即项目根目录）
# 结果路径：llm_from_scratch/part_3
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402  # ← flake8 会报 E402（import 不在文件顶部），这里明确忽略

# ==========================================
# BPE 分词器：Part 4 的核心组件
# ==========================================
# tokenizer_bpe.py 实现了字节对编码（Byte-Pair Encoding）分词器，
# 它是 GPT-2/LLaMA 等现代 LLM 的标准分词方式。
# 与 Part 3 的 ByteTokenizer（简单按字节切分）不同，BPE 通过统计学习出
# 高频子词组合，用更少的 token 表示更多文本，提高模型的"信息密度"。
from tokenizer_bpe import BPETokenizer


# ==========================================
# main()：脚本的入口函数
# ==========================================
# 负责解析参数、加载模型、执行生成、输出结果。
# 整个函数按"数据流"顺序组织：参数 → 设备 → 检查点 → 分词器 → 配置 → 模型 → 生成 → 解码。
def main():
    # ─── 1. 命令行参数解析 ───
    # argparse 是 Python 标准库的命令行参数解析器。
    # ArgumentParser 自动生成 --help 帮助信息，处理参数类型转换和验证。
    p = argparse.ArgumentParser()
    #   --ckpt：检查点文件路径（必填），包含模型权重和配置。
    p.add_argument('--ckpt', type=str, required=True)
    #   --prompt：用户输入的提示文本（可选），模型将从此文本开始续写。
    #            默认为空字符串，模型会从 BOS token（ID=10）开始自由生成。
    p.add_argument('--prompt', type=str, default='')
    #   --tokens：最多生成多少个新 token，默认 100。
    #            模型达到此数量或遇到 EOS 时会停止。
    p.add_argument('--tokens', type=int, default=100)
    #   --cpu：强制使用 CPU 推理（即使有 GPU）。
    #          `action='store_true'` 表示这是一个开关标志——加了 --cpu 就设为 True，不加就默认 False。
    p.add_argument('--cpu', action='store_true')
    # 语法：p.parse_args() 解析 sys.argv（命令行参数），返回一个 Namespace 对象，
    #       其属性名就是 --xxx 去掉前缀后的名字，如 args.ckpt、args.prompt。
    args = p.parse_args()

    # ─── 2. 设备选择 ───
    # 语法：`A if 条件 else B` 是 Python 三元表达式（内联 if-else）。
    # torch.cuda.is_available() 检查 PyTorch 是否能访问 NVIDIA GPU。
    # 逻辑：有 GPU 且用户没加 --cpu → 用 cuda；否则 → 用 cpu。
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ─── 3. 加载检查点 ───
    # torch.load() 用 pickle 反序列化 .pt 文件。
    # map_location='cpu' 强制权重先加载到 CPU——这是安全做法，
    # 避免 GPU 显存不足或不同设备间权重张量的设备不匹配。
    ckpt = torch.load(args.ckpt, map_location='cpu')  # 先在 CPU 加载；后续再把模型整体搬到目标设备
    # 检查点文件通常是一个 dict，包含：
    #   ckpt['model']  → 模型的 state_dict（OrderedDict，键=参数名，值=权重张量）
    #   ckpt['config'] → 模型架构配置（可能不存在，此时需从权重推断）
    sd = ckpt['model']
    # 语法：dict.get(key) 与 dict[key] 的区别——
    #   dict[key] 在 key 不存在时抛出 KeyError；
    #   dict.get(key) 在 key 不存在时返回 None，不报错。
    # 这里用 get 容错，旧检查点可能没有存 config。
    cfg = ckpt.get('config') or {}  # 若 config 为 None/不存在，用空字典兜底

    # ─── 4. 加载 BPE 分词器 ───
    # 分词器是独立于模型权重保存的（因为它是文本预处理工具，不是神经网络的参数）。
    # 我们在检查点旁边放一个 tokenizer_dir.txt 文件，里面记录分词器目录的路径。
    tok = None
    # 语法：Path(args.ckpt).with_name('tokenizer_dir.txt')
    #   with_name() 把文件名替换为 'tokenizer_dir.txt'，但保留父目录不变。
    #   例如 args.ckpt='checkpoints/ckpt.pt' → Path('checkpoints/tokenizer_dir.txt')
    tok_dir_file = Path(args.ckpt).with_name('tokenizer_dir.txt')
    if tok_dir_file.exists():
        # read_text() 读取整个文件为字符串，strip() 去除首尾空白和换行。
        tok_dir = tok_dir_file.read_text().strip()  # 文件内容为分词器目录路径
        tok = BPETokenizer()
        # 从目录中加载 merges.txt 和 vocab.json 两个文件。
        # BPETokenizer.load() 会读取合并规则表和词表，重建完整的编码/解码能力。
        tok.load(tok_dir)                            # ← 实例方法，传入目录路径
        # 从分词器获取词表大小——这个值必须与模型输出层的 vocab_size 一致，
        # 否则 logits 的维度与分词器 ID 范围不匹配，无法正确解码。
        vocab_from_tok = tok.vocab_size
    else:
        # 没有分词器，后续用原始 UTF-8 字节做"退化"编解码（每个字节当作一个 token）。
        vocab_from_tok = None


    # ─── 5. 构建模型配置 ───
    # 设计决策：优先使用检查点保存的 config，没有的话从权重形状反向推断。
    # 为什么要反向推断？Part 3 训练时可能没存 config；但权重的形状天然编码了
    # 架构信息——例如 tok_emb.weight 的形状 [vocab_size, n_embd] 暴露了词表和隐层维度。
    if not cfg:
        # ═══════════════════════════════════════
        # 情形 A：有 config → 直接用（已有 cfg，不需要额外操作）
        # ═══════════════════════════════════════
        # 注意：这里原本有处理"分词器词表比检查点大"的逻辑（已注释掉），
        # 用于支持"扩展词表"场景（如添加特殊 token 后微调）。
        # If a tokenizer is present and vocab differs, override with tokenizer vocab
        # if vocab_from_tok is not None and cfg.get('vocab_size') != vocab_from_tok:
        #     cfg = {**cfg, 'vocab_size': vocab_from_tok}
    # else:
        # ═══════════════════════════════════════
        # 情形 B：无 config → 从权重形状反向推断配置
        # ═══════════════════════════════════════
        # 这段逻辑处理"老检查点"——它们没有保存 cfg dict，
        # 但权重的形状天然暴露了模型架构信息。

        # ── 5a. 推断 vocab_size 和 n_embd ──
        # tok_emb.weight 是词嵌入矩阵，形状为 [vocab_size, n_embd]。
        # 语法：`V, C = sd['tok_emb.weight'].shape` 是元组解包——
        #   shape 返回 (vocab_size, n_embd)，直接赋值给两个变量。
        #   V = 词表大小，C = 隐层维度（n_embd）。
        V, C = sd['tok_emb.weight'].shape

        # ── 5b. 推断 block_size ──
        # pos_emb.weight 是位置嵌入表，形状为 [block_size, n_embd]。
        # 每个位置有一个 C 维向量，所以行数就是最大上下文长度。
        # 但 Part 3 模型可能用 RoPE 替代了位置嵌入——此时 pos_emb 不存在，
        # 只能给一个保守的默认值 256。
        # 语法：`A if 条件 else B`（三元表达式），`'key' in dict` 检查键是否存在。
        block_size = sd['pos_emb.weight'].shape[0] if 'pos_emb.weight' in sd else 256

        # ── 5c. 推断 n_layer ──
        # Transformer Block 的参数键名格式为 "blocks.0.attn.q_proj.weight"。
        # 用正则提取出 block 编号，找最大值 + 1 即为总层数。
        import re
        # 语法：`(m := re.match(...))` 是 Python 3.8+ 的海象运算符（walrus operator），
        #   在 if 条件中做正则匹配的同时把结果赋给 m，避免"先匹配再判断"的两行写法。
        # 集合推导式：遍历所有权重键名 → 正则匹配 "blocks.数字." → 提取数字 → 去重。
        layer_ids = {int(m.group(1)) for k in sd.keys() if (m := re.match(r"blocks\.(\d+)\.", k))}
        # max() + 1：如果最大编号是 7，说明有 blocks.0 ~ blocks.7，共 8 层。
        n_layer = max(layer_ids) + 1 if layer_ids else 1

        # ── 5d. 推断 n_head ──
        # 头数不影响权重的形状（QKV 投影合并为一个大矩阵），无法从权重直接推断。
        # 策略：选一个能整除 C（n_embd）的值，优先选 8（最常见的选择）。
        #   能被 8 整除 → 8 头；能被 4 整除 → 4 头；能被 2 整除 → 2 头；否则 1 头。
        n_head = 8 if C % 8 == 0 else 4 if C % 4 == 0 else 2 if C % 2 == 0 else 1

        # ── 5e. 组装完整配置字典 ──
        # 优先用分词器词表大小（它反映了训练时的真实词表），
        # 没有分词器时退回到权重的嵌入行数 V。
        cfg = dict(
            vocab_size=vocab_from_tok or V,   # `or` 短路逻辑：vocab_from_tok 为 None 时取 V
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=C,
            dropout=0.0,          # 推理阶段不需要 dropout，设为 0
            use_rmsnorm=True,     # Part 3 默认使用 RMSNorm（假设旧模型也是）
            use_swiglu=True,      # Part 3 默认使用 SwiGLU 激活
            rope=True,            # Part 3 默认使用 RoPE 位置编码
            max_pos=4096,         # RoPE 最大位置编码范围
            sliding_window=None,  # 不限制注意力窗口（全局注意力）
            attention_sink=0,     # 不保留注意力水槽 token
        )

    # ─── 6. 构建并加载模型 ───
    # 语法：`**cfg` 是字典解包（dictionary unpacking），
    #   把 cfg 的键值对展开为关键字参数传递给 GPTModern 构造函数。
    #   等价于 GPTModern(vocab_size=..., block_size=..., n_layer=..., ...)。
    model = GPTModern(**cfg).to(device).eval()
    # load_state_dict() 把检查点中的权重张量逐个复制到模型的对应参数上。
    # 注意：这里假设检查点的参数名与模型定义完全一致——如果不一致会报错。
    model.load_state_dict(ckpt['model'])
    # 再次调用 .to(device) 和 .eval() 确保模型在正确设备和评估模式下。
    # .eval() 的作用：关闭 Dropout（训练时随机丢弃神经元，推理时全保留），
    #   并让 BatchNorm 等层使用训练时统计的全局均值/方差。
    model.to(device).eval()

    # ─── 7. 将 prompt 编码为 token ID 序列 ───
    # 编码方式取决于是否有分词器：
    #   有 BPE 分词器 → 用 BPE 算法切分子词，得到高频子词 ID 序列
    #   无分词器     → 退化方案：每个 UTF-8 字节当作一个 token ID
    if tok:
        # 语法：tok.encode(text) 调用 BPE 分词器，把文本转为 token ID 列表。
        ids = tok.encode(args.prompt)
        # 如果 prompt 为空或编码后为空序列，至少送入一个 BOS token（ID=10），
        # 让模型有一个起点可以开始生成。
        if len(ids) == 0: ids = [10]
    else:
        # 退化方案：直接用 UTF-8 编码。中文等多字节字符会变成多个字节 ID，
        # 模型需要自己学会组合它们——效果通常很差，但至少不会崩溃。
        ids = [10] if args.prompt == '' else list(args.prompt.encode('utf-8'))

    # 语法：torch.tensor([ids]) 将 list 转为二维张量。
    #   dtype=torch.long 指定元素为长整型（token ID 必须是整数）。
    #   device=device 直接在目标设备上创建张量，避免 CPU→GPU 的传输开销。
    # 形状：(1, T) —— batch_size=1（单条 prompt），T = prompt 的 token 数。
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # ─── 8. 自回归文本生成 ───
    # 语法：`torch.no_grad()` 上下文管理器。
    #   在 no_grad 区域内，PyTorch 不会构建计算图、不记录梯度。
    #   为什么要用？推理时不需要反向传播，禁用 autograd 可以：
    #     a) 大幅减少显存占用（不存中间激活值）；
    #     b) 加快计算速度（跳过梯度相关的簿记开销）。
    #   这是所有推理代码的标准做法。
    with torch.no_grad():
        # model.generate() 是 GPTModern 的自回归生成方法。
        # 内部流程：
        #   第一阶段（prefill）：把整个 prompt 编码，一次性计算 K/V 并缓存；
        #   第二阶段（decode）：逐个生成新 token，每次只算最新 token 的 K/V。
        # 参数 max_new_tokens 控制最多生成多少新 token（不含 prompt 部分）。
        out = model.generate(idx, max_new_tokens=args.tokens)

    # ─── 9. 解码并输出生成的文本 ───
    # 语法：out[0] 取 batch 中的第一条（也是唯一一条）序列，
    #   .tolist() 把 GPU 张量转为 Python 整数列表。
    out_ids = out[0].tolist()
    if tok:
        # BPE 解码：把 token ID 序列还原为人类可读的文本。
        # 注意：解码出的文本包含原始 prompt + 模型续写部分。
        print(tok.decode(out_ids))
    else:
        # 退化解码：把整数 ID 列表当作 UTF-8 字节序列解释。
        # errors='ignore' 忽略无效字节序列（如不完整的 UTF-8 编码），
        # 防止因单个解码错误导致整段输出崩溃。
        print(bytes(out_ids).decode('utf-8', errors='ignore'))

# 语法：`if __name__ == '__main__':` 是 Python 的"脚本入口"保护。
#   当文件被直接运行（python sample.py）时，__name__ 为 '__main__'，执行 main()；
#   当文件被 import 时，__name__ 为模块名 'sample'，不执行 main()。
#   这样既可以用作命令行工具，也可以被其他脚本引用而不自动运行。
if __name__ == '__main__':
    main()

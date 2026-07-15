# ==========================================
# Part 6.6：SFT 模型推理 — 加载 checkpoint，从指令生成回答
# ==========================================
# 本脚本是 SFT 训练流程的最后一步：验证微调效果。
# 加载训练好的 checkpoint，给定一条用户指令（prompt），让模型生成回答。
#
# 推理流程概览（端到端）：
#   1. 加载 SFT checkpoint（模型权重 + 配置）
#   2. 用与训练时相同的 tokenizer 和 formatter 处理输入 prompt
#   3. 调用 model.generate() 自回归生成 token 序列
#   4. 将生成的 token ID 解码回人类可读文本
#
# 与 Part 4 预训练采样的关键区别：
#   - 预训练采样：给一句开头，模型"续写"下去（故事/代码/文章）
#   - SFT 采样：给一条指令，模型"回答"这个指令（问答/翻译/代码改写）
#   - 核心差异在 prompt 格式——SFT 需要用对话模板包装，让模型识别
#     "现在是 User 在说话，该我来回答了" 的角色切换信号

from __future__ import annotations
import argparse, torch

# ==========================================
# 导入 Part 3 的 GPTModern 模型
# ==========================================
# 推理时复用与训练相同的模型代码，确保架构完全一致。
# 模型内部已包含 KV Cache 优化（Part 3 实现），自回归生成时
# 每一步只需计算最新 token 的 K/V，历史 token 的 K/V 直接复用。
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402

# ─── 导入 Part 6 的本地组件 ───
# SFTCollator：为了复用它的 tokenizer（编码 prompt + 解码生成的 token）
# format_prompt_only：用训练时间相同的对话模板包装用户输入
from collator_sft import SFTCollator
from formatters import format_prompt_only


# ==========================================
# main()：SFT 推理主流程
# ==========================================
def main():
    # ─── 命令行参数 ───
    p = argparse.ArgumentParser()

    # --ckpt：SFT 训练产出的 checkpoint 路径（必需参数）。
    # 语法：required=True 表示此参数必传，不传则 argparse 自动报错并显示帮助。
    p.add_argument('--ckpt', type=str, required=True)

    # --prompt：用户输入的指令文本（必需参数）。
    # 例如 "What are the three primary colors?" 或一段待改写的代码。
    p.add_argument('--prompt', type=str, required=True)

    # ─── 模型架构参数 ───
    # 这些值必须与训练时保持一致！虽然 checkpoint 里的 config 记录了架构信息，
    # 但为了简洁，这里用命令行参数显式指定（也可改为从 cfg 自动读取）。
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)

    # --tokens：最大生成 token 数。模型生成这么多 token 后自动停止。
    # 80 对于大多数问答来说足够；过长的回答会被截断。
    p.add_argument('--tokens', type=int, default=80)

    # --temperature：采样温度，控制生成的随机性。
    #   0.0  → 贪心解码（greedy），每次都选最高概率的 token，输出完全确定
    #   0.2  → 低温度，输出较确定但保留少量多样性（SFT 的常用值）
    #   1.0  → 原样概率分布，输出较随机、有创意
    #   >1.0 → 高温度，输出非常随机，可能产生"幻觉"或无意义内容
    # 比喻：temperature 像"创意开关"——0 是照本宣科，1 是自由发挥，0.2 是稳健偏保守。
    p.add_argument('--temperature', type=float, default=0.2)

    # --cpu：强制 CPU 推理（不要求有 GPU）
    p.add_argument('--cpu', action='store_true')

    # --bpe_dir：BPE tokenizer 路径，用于编码 prompt 和解码输出
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer')

    args = p.parse_args()

    # ─── 设备选择 ───
    # 语法：`A if 条件 else B` 三元表达式选择设备。
    # 推理通常比训练快得多，即使 CPU 上也能在几秒内完成。
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ==========================================
    # 第一步：加载 SFT Checkpoint
    # ==========================================
    # torch.load() 反序列化 pickle 文件，返回保存时的 Python 字典。
    # map_location=device 让张量在加载时直接放到目标设备，
    # 避免先加载到 CPU 再 .to(device) 的额外开销。
    ckpt = torch.load(args.ckpt, map_location=device)

    # cfg 是训练时保存的模型配置字典，记录了架构参数（n_layer、vocab_size 等）。
    # 这里从 checkpoint 读取 vs 用命令行参数：两种方式都可行，
    # 当前代码混合使用（block_size 从 cfg 读，其他从 args 读）。
    cfg = ckpt.get('config', {})

    # ==========================================
    # 第二步：构建 Tokenizer 和模型
    # ==========================================
    # SFTCollator 内部管理 tokenizer（BPE 或 Byte-level），
    # 这里复用 collator 只是为了用它的 tokenizer 来编码和解码。
    # block_size 从 checkpoint 配置读取（而非命令行），确保与训练时一致。
    col = SFTCollator(block_size=cfg.get('block_size', 256), bpe_dir=args.bpe_dir)

    # 用与训练时完全相同的架构参数初始化模型。
    # 注意：这些参数（n_layer, n_head, n_embd）必须与 checkpoint 中的权重匹配！
    # 如果不一致，load_state_dict 会因参数名/形状不匹配而报错。
    model = GPTModern(vocab_size=col.vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)

    # 将 checkpoint 中的权重"安装"到刚创建的模型上。
    # load_state_dict 按参数名严格匹配，不匹配会抛出异常——这确保权重不会错位。
    model.load_state_dict(ckpt['model'])

    # ─── 切换到评估模式 ───
    # model.eval() 关闭训练专用行为：
    #   - Dropout 层变为恒等映射（不再随机丢弃神经元）
    #   - BatchNorm（如果有）使用训练时积累的全局统计量而非 batch 统计量
    # 推理时必须调用 eval()，否则 Dropout 会随机置零部分输出，导致结果不稳定。
    model.eval()

    # ==========================================
    # 第三步：格式化 + 编码 Prompt
    # ==========================================
    # 用训练时相同的对话模板包装用户输入的指令。
    # 例如 prompt="三原色是什么？" →
    #   format_prompt_only 输出 "User: 三原色是什么？\nAssistant: "
    # 这样模型看到的上下文与训练时完全一致，能正确识别
    # "现在该 Assistant 说话了"的角色切换。
    prompt_text = format_prompt_only(args.prompt).replace('</s>', '')

    # 将格式化后的文本编码为 token ID 序列。
    # col.encode() 返回 List[int]，如 "User: What is DNA?\nAssistant: " →
    #   [85, 117, 115, 101, 114, 58, 32, ...]
    ids = col.encode(prompt_text)

    # 语法：torch.tensor([ids], dtype=..., device=...) 将列表包装为二维张量。
    # 注意 [ids] 外面多一层方括号——这是为了创建 batch 维度：
    #   ids = [1, 2, 3]           → Python 列表，长度 = seq_len
    #   [ids] = [[1, 2, 3]]       → 嵌套列表，表示 batch_size=1
    #   torch.tensor([ids])       → 张量形状 (1, T)，即 batch_size=1, seq_len=T
    # dtype=torch.long 是因为 token ID 是整数索引（64 位有符号整数）
    # device=device 让数据直接放到 GPU/CPU 上，与模型在同一设备
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # ==========================================
    # 第四步：自回归生成
    # ==========================================
    # 语法：@torch.no_grad() 是 PyTorch 的上下文管理器装饰器，
    # 进入 with 块后，所有操作都不会构建计算图、不追踪梯度。
    # 这带来两个好处：
    #   1. 显存大幅减少（不需要存储中间激活值用于反向传播）
    #   2. 计算更快（跳过梯度计算的额外开销）
    # 推理时必须用 no_grad()，否则显存会快速爆炸。
    with torch.no_grad():
        # model.generate() 是 GPTModern 中实现的自回归生成方法（Part 3）。
        # 参数说明：
        #   idx：输入的 token ID 张量，形状 (1, prompt_len)
        #   max_new_tokens：最多生成多少个新 token
        #   temperature：采样温度，控制随机性（0=贪心，>0=按概率采样）
        #   top_k：只从概率最高的 k 个 token 中采样（这里是 top-3），
        #          过滤掉低概率的"噪音" token，提升生成质量。
        #          类比：top_k 像"初选"——先选出 3 个最佳候选，再用 temperature 做"决赛"。
        #
        # 生成过程内部使用 KV Cache：
        #   第一步（prefill）：将整个 prompt 喂入模型，计算所有层的 K/V 并缓存。
        #   后续步（decode）：每步只计算最新 token，历史 K/V 从缓存读取，
        #                   计算量从 O(T²) 降至 O(T)，推理大幅加速。
        out = model.generate(idx, max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=3)

    # ==========================================
    # 第五步：解码输出
    # ==========================================
    # out 形状 (1, prompt_len + generated_len)，包含完整序列：
    #   out[0:prompt_len]  → prompt 的 token ID（输入）
    #   out[prompt_len:]   → 模型生成的 token ID（输出）
    #
    # 语法：out[0].tolist() 取第一个 batch（batch_size=1），转为 Python 列表。
    # 语法：.tolist() 将 PyTorch 张量转为 Python list，比逐个 .item() 更高效。
    out_ids = out[0].tolist()

    # 记住原始 prompt 的 token 数量，用于截取"只生成的部分"。
    # 语法：idx.size(1) 取 idx 张量的第 1 维（时间维/序列长度维）的大小。
    # idx 形状 (1, T) → idx.size(1) = T = prompt 的 token 数
    orig_len = idx.size(1)

    # ─── 解码策略：优先用 BPE tokenizer 解码完整文本 ───
    # hasattr 检查 collator 的 tokenizer 是否有 decode 方法。
    # BPE tokenizer 有 decode（逆向查表），Byte tokenizer 需要手动 bytes 解码。
    if hasattr(col, "tok") and hasattr(col.tok, "decode"):
        # BPE 路径：用训练好的 tokenizer 将 token ID 序列解码回文本。
        # decode full text or just the generated suffix; suffix is often clearer
        # 解码整个序列（prompt + 生成内容），让用户看到完整的对话上下文。
        generated = col.tok.decode(out_ids)
        print(generated)
    else:
        # Byte-level 路径：只解码生成的 token ID（跳过 prompt 部分）。
        # 语法：out_ids[orig_len:] 切片，从 prompt 末尾开始取（只取生成的部分）。
        # bytes(integer_list) 将 0-255 的整数列表转为 bytes 对象。
        # .decode("utf-8", errors="ignore") 将 UTF-8 字节解码为字符串，
        #   errors="ignore" 跳过无法解码的字节（如不完整的 UTF-8 序列），
        #   避免因一个坏字节导致整个解码失败。
        generated = bytes(out_ids[orig_len:]).decode("utf-8", errors="ignore")
        print(generated)


# ==========================================
# 脚本入口
# ==========================================
# 语法：`if __name__ == '__main__'` 确保仅在直接运行时执行 main()，
# 被 import 时不执行。这是 Python CLI 脚本的标准模式。
if __name__ == '__main__':
    main()

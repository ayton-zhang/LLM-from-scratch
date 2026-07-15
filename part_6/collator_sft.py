# ==========================================
# Part 6.2：SFT 数据整理器 — token 化 + Label Masking
# ==========================================
# 本模块是 SFT 训练流程中最关键的"数据预处理"环节。
# 核心任务：将人类可读的 (prompt, response) 文本对，转换为模型能消费的
# 数字序列（token IDs），并在 label 中对 prompt 部分做屏蔽处理。
#
# 为什么需要 Label Masking（标签屏蔽）？
#   在 SFT 中，一条训练样本的结构是：
#     "User: 三原色是什么？\nAssistant: 红、黄、蓝。"
#   模型应该学习的是"给定 User 指令，生成 Assistant 的回答内容"。
#   如果让模型也对"User: 三原色是什么？\nAssistant: "这部分计算 loss，
#   等于逼它"背诵用户的问题和格式标记"，这完全浪费算力，还可能引入偏差。
#
#   因此 SFT 的标准做法是：将 prompt 区域（包括格式标记）的 label 设为 -100，
#   PyTorch 的 CrossEntropyLoss 遇到 ignore_index=-100（默认值）时自动跳过，
#   只有 response 区域真正参与梯度更新。
#
# 数据流摘要：
#   (prompt, response) 文本对
#     → 用 formatter 组装成完整对话文本
#     → tokenizer 编码为整数 ID 序列
#     → 做 causal LM 的 label shift（y[t] = x[t+1]，预测下一个 token）
#     → 将 prompt 区域的 label 置为 -100（屏蔽 loss）
#     → 截断/填充到 block_size 统一长度
#     → 输出 (xb, yb) 张量对，直接喂给模型

from __future__ import annotations
from typing import List, Tuple
import torch
import traceback

# ==========================================
# Tokenizer 加载：BPE 优先 → Byte-level 兜底
# ==========================================
# SFT 训练必须使用与预训练阶段完全相同的 tokenizer！
# 因为预训练模型的 embedding 矩阵和 lm_head 的维度由 tokenizer 的 vocab_size 决定，
# 如果 tokenizer 不匹配，token ID 映射关系完全不同，预训练权重将完全无效。
#
# 加载策略（优先级递减）：
#   1. BPE Tokenizer（Part 4 训练出的）——与预训练模型匹配，首选
#   2. Byte Tokenizer（Part 3 的 256 字节级）——无需训练，但 vocab 太小，效果差
#   3. 都不可用时抛出 RuntimeError——没有 tokenizer 没法训练

import sys
from pathlib import Path as _P

# 语法：sys.path.append(...) 将 part_4/ 目录加入模块搜索路径。
# .parents[1] 取当前文件的父目录的父目录（即 llm_from_scratch/），
# 然后拼接 'part_4' 形成完整路径。这样 `from tokenizer_bpe import BPETokenizer` 能正确解析。
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))

# ─── 尝试加载 BPE Tokenizer（Part 4 训练的） ───
# BPE（Byte Pair Encoding）是现代 LLM 最常用的分词算法。
# 它通过统计字符对的共现频率，迭代合并高频对来构建词汇表，
# 能有效处理罕见词（拆分为子词）和常见词（保留为完整 token）。
# 相比字节级 tokenizer（每个字节一个 ID），BPE 的序列更短、语义密度更高。
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True          # 标记 BPE 可用，供后续 __init__ 判断
except Exception:
    _HAS_BPE = False          # BPE 导入失败（如缺少 tokenizers 库），降级到字节级

# ─── 尝试加载 Byte Tokenizer（Part 3 的简单实现） ───
# 字节级 tokenizer 直接将 UTF-8 编码的每个字节映射为 0-255 的 token ID。
# 优点：零训练成本、vocab 固定 256、能处理任何语言（所有 Unicode 字符都能编码）
# 缺点：序列极长（一个中文字符 = 3 字节 = 3 个 token），效率低
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None      # 设为 None 而不是崩溃，统一在 __init__ 中处理异常

# ─── 导入 formatters：负责将原始文本组装为对话格式 ───
# formatters.py 定义了 Example 数据类和 format_example/format_prompt_only 函数，
# 它们负责将 (prompt, response) 原始文本包装成统一的对话模板格式，例如：
#   "User: {prompt}\nAssistant: {response}"
# 这种模板化的好处是：模型在推理时能识别 "User:" 和 "Assistant:" 标记，
# 知道何时该自己说话、何时该等用户输入。
from formatters import Example, format_example, format_prompt_only


# ==========================================
# SFTCollator：SFT 数据整理器
# ==========================================
# 职责：将一批 (prompt, response) 文本对转换为模型可消费的张量格式。
# 类比：这个类就像食堂的"打菜师傅"——原始食材（文本）经过他的处理，
# 变成标准化的餐盘（张量），模型（食客）直接拿着就能吃，不用操心格式问题。
#
# 核心功能：
#   1. Tokenizer 管理：自动选择最佳可用的 tokenizer（BPE > Byte > 报错）
#   2. 文本格式化：用对话模板包装 prompt 和 response
#   3. Token 化：将格式化后的文本编码为整数 ID 序列
#   4. Label 构建：做 causal LM 的 shift 操作 + prompt 区域 masking
#   5. 长度统一：截断过长序列、填充过短序列至 block_size
class SFTCollator:
    """Turn (instruction,response) into token ids and masked labels for causal LM (6.2).
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """

    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        # block_size：最大序列长度（token 数）。
        # 超过此长度的序列会被截断，不足的会被填充。这是 Transformer 的固定上下文窗口大小。
        self.block_size = block_size

        # ─── Tokenizer 初始化：三级降级策略 ───
        # 设计思路：在多种运行环境（有网络/无网络、有 BPE/无 BPE）下都能工作，
        # 但能力随环境降级（BPE → Byte → 报错）。
        self.tok = None

        # 第一优先级：BPE Tokenizer（与预训练模型匹配，效果最好）
        if _HAS_BPE:
            # If a trained tokenizer directory exists from Part 4, you can `load` it.
            # Otherwise we create an ad-hoc BPE on the fly using fallback prompts during demo.
            try:
                # 创建一个 vocab_size=8000 的 BPE tokenizer 实例。
                # 8000 是 Part 4 训练 BPE 时使用的词汇表大小。
                # 注意：仅仅创建实例还不够——必须加载训练好的 merge 规则和 vocab，
                # 否则它只是个"空壳"，不知道如何分词。
                self.tok = BPETokenizer(vocab_size=8000)

                if bpe_dir:
                    # .load(bpe_dir) 从磁盘加载预先训练好的 BPE 合并规则和词汇表。
                    # bpe_dir 通常指向 part_4/runs/part4-demo/tokenizer/，
                    # 其中包含 merges.txt（BPE 合并规则）和 vocab.json（词汇表映射）。
                    self.tok.load(bpe_dir)
                    print(f"Loaded BPE tokenizer from {bpe_dir}")
                else:
                    # 如果 bpe_dir 为 None（未指定路径），BPE tokenizer 处于"未训练"状态。
                    # 在实际 demo 中，orchestrator.py 会传入 --bpe_dir，所以这里不会走到。
                    # weak ad-hoc training would belong elsewhere; for the demo we assume Part 4 tokenizer exists
                    pass
            except Exception:
                # 加载失败时打印完整堆栈便于调试，然后将 tok 设为 None，
                # 触发下面的第二优先级策略（Byte Tokenizer）。
                print(traceback.format_exc())
                self.tok = None

        # 第二优先级：Byte Tokenizer（无需训练，vocab=256，万能兜底）
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()

        # 第三优先级：彻底失败——没有 tokenizer 就没法训练
        # 语法：raise RuntimeError("msg") 抛出一个运行时异常，终止程序。
        # 这里选择报错而非静默失败，因为 tokenizer 是训练的必要条件。
        if self.tok is None:
            raise RuntimeError(
                "No tokenizer available. "
                "Install tokenizers or ensure Part 3 ByteTokenizer exists."
            )

    # ==========================================
    # vocab_size 属性：获取 tokenizer 的词汇表大小
    # ==========================================
    # 语法：@property 将方法变为"属性"——调用时不用写括号：
    #   col.vocab_size  （有 @property）
    #   而非 col.vocab_size()  （无 @property）
    # 这样更符合"词汇表大小是 collator 的属性"的直觉。
    #
    # 语法：getattr(obj, 'attr', default) 安全获取属性值：
    #   如果 obj 有 'vocab_size' 属性 → 返回其值
    #   如果没有（如某些自定义 tokenizer 遗漏了这个属性）→ 返回默认值 256
    #   256 = 2^8，是 ByteTokenizer 的标准词汇表大小（0-255 覆盖所有字节值）
    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    # ==========================================
    # encode()：将文本编码为 token ID 列表
    # ==========================================
    # 不同的 tokenizer 可能有不同的返回类型（list、torch.Tensor 等），
    # 这里统一转换为 Python list[int]，保证下游 collate 逻辑的一致性。
    def encode(self, text: str) -> List[int]:
        # 语法：hasattr(obj, 'name') 检查对象是否有某个属性/方法。
        # BPE Tokenizer 有 .encode() 方法；ByteTokenizer 可能没有，
        # 此时走 else 分支的 UTF-8 字节编码。
        if hasattr(self.tok, 'encode'):
            # 调用 tokenizer 的 encode 方法，将文本转为 token ID 序列
            ids = self.tok.encode(text)

            # 统一类型：如果是 PyTorch 张量，转为 Python 列表。
            # 语法：isinstance(obj, Type) 检查对象是否是某类型的实例。
            # .tolist() 是 PyTorch 张量的方法，将张量转为 Python 列表。
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids

        # ByteTokenizer-like：直接对文本做 UTF-8 编码。
        # text.encode('utf-8') 返回 bytes 对象，如 b'Hello' → [72, 101, 108, 108, 111]。
        # 语法：list(bytes_object) 将 bytes 迭代为整数列表，每个整数在 0-255 范围内。
        # 这本质上就是 ByteTokenizer 的实现原理——每个字节一个 ID。
        return list(text.encode('utf-8'))

    # ==========================================
    # collate()：核心数据整理方法
    # ==========================================
    # 这是 SFTCollator 的核心方法，将一批 (prompt, response) 文本对
    # 转换为两个张量 (xb, yb)，分别作为模型的输入和训练目标。
    #
    # 参数：
    #   batch: List[Tuple[str, str]]，每个元素是 (prompt, response) 文本对
    #
    # 返回：
    #   xb: torch.Tensor，形状 (B, T)，输入 token ID 序列
    #   yb: torch.Tensor，形状 (B, T)，目标 token ID 序列，prompt 区域为 -100
    #
    # 其中 B = batch_size, T = block_size
    def collate(self, batch: List[Tuple[str, str]]):
        # Build "prompt + response" and create label mask where prompt positions are -100.

        input_ids = []   # 存放每条样本的输入 token ID 列表
        labels = []      # 存放每条样本的 label token ID 列表（prompt 区域为 -100）

        # ─── 逐样本处理：格式化 → 编码 → shift → masking ───
        # 这不是批量操作，因为每条样本长度不同，必须先各自处理再统一填充。
        for prompt, response in batch:
            # ---------- 步骤 1：文本格式化 ----------
            # format_prompt_only(prompt) 只生成 prompt 侧的格式化文本。
            # 例如输入 "三原色是什么？" → 输出 "User: 三原色是什么？\nAssistant: "
            # .replace('</s>', '') 去掉可能存在的结束标记（</s> 是某些 tokenizer 的特殊 token）
            prefix_text = format_prompt_only(prompt).replace('</s>', '')

            # format_example(Example(prompt, response)) 生成完整的对话格式化文本。
            # 例如 Example("三原色是什么？", "红黄蓝") →
            #   "User: 三原色是什么？\nAssistant: 红黄蓝"
            # 这个完整文本将作为模型的输入，模型看完 User 问题后接续生成 Assistant 回答。
            text = format_example(Example(prompt, response))

            # ---------- 步骤 2：Token 化 ----------
            # 将格式化后的文本编码为 token ID 序列，并截断到 block_size。
            # 语法：[:self.block_size] 是 Python 切片语法，只保留前 block_size 个元素。
            # 如果序列本就不足 block_size，切片不做任何事（不会补零）。
            ids = self.encode(text)[:self.block_size]
            # 单独编码 prompt 前缀，用于确定"哪些位置属于 prompt"。
            # 截断到 block_size 是为了与 ids 对齐——如果 prompt 本身就超长，
            # prefix_text 编码后也会被截断，n_prompt 会等于 len(ids)。
            prompt_ids = self.encode(prefix_text)[:self.block_size]

            # 确定 prompt 部分的 token 数量。
            # 为什么用 min(len(prompt_ids), len(ids)) 而不是 len(prompt_ids)？
            #   如果 prompt 比 block_size 还长，prompt_ids 和 ids 都被截到 block_size，
            #   此时 len(prompt_ids) == len(ids)，min 两者相同。
            #   如果 response 部分的 token 被截掉了（ids 截断了但 prompt_ids 完整），
            #   len(prompt_ids) 可能 > len(ids)，此时取 len(ids) 作为 mask 范围。
            n_prompt = min(len(prompt_ids), len(ids))

            # ---------- 步骤 3：构建 Causal LM 的 Label Shift ----------
            # 自回归语言模型的核心任务：给定前 t 个 token，预测第 t+1 个 token。
            # 因此 label 需要对 input 做一位偏移：
            #   x = [t0, t1, t2, t3, ...]        # 模型输入（"到目前为止的序列"）
            #   y = [t1, t2, t3, ..., -100]       # 训练目标（"下一步该是什么"）
            #
            # 具体操作：
            #   y[t] = x[t+1]  →  位置 t 的目标是预测下一个位置的 token
            #   y[-1] = -100   →  最后一个位置没有"下一个 token"，设为忽略
            x = ids                          # 输入序列

            # 语法：list.copy() 做浅拷贝（shallow copy）。对于整数列表，
            # 浅拷贝和深拷贝效果相同——整数是不可变对象。
            # 必须 copy 而非直接 y = ids，否则修改 y 也会影响 x。
            y = ids.copy()

            # 对每个位置 t（除了最后一个），让它预测 t+1 位置的实际 token
            for t in range(len(y) - 1):     # 只遍历到倒数第二个位置
                y[t] = ids[t + 1]           # 目标 = 下一个 token
            # 最后一个位置没有"下一个 token"可预测，设为 -100（忽略）
            y[-1] = -100

            # ---------- 步骤 4：屏蔽 Prompt 区域的 Label ----------
            # 这是 SFT 与预训练最大的区别！
            # 预训练时所有位置都参与 loss 计算（每个 token 都要被预测）。
            # SFT 时 prompt 区域的 label 被设为 -100，PyTorch 的 CrossEntropyLoss
            # 默认 ignore_index=-100，自动跳过这些位置，loss 只来自 response 区域。
            #
            # 为什么是 range(n_prompt-1) 而不是 range(n_prompt)？
            #   n_prompt 个 prompt token 占据了位置 0 到 n_prompt-1。
            #   但 y[t] 存的是 ids[t+1]（已经做过 shift 了），所以：
            #     y[0]   = ids[1]  → 第 1 个 prompt token 预测第 2 个 prompt token
            #     y[n_prompt-2] = ids[n_prompt-1] → 第 n_prompt-1 个 prompt token 预测第 n_prompt 个
            #   位置 n_prompt-1 存的是 ids[n_prompt]，即 response 的第一个 token，
            #   它虽然在"prompt 区域索引"内，但预测内容是 response 的开头——我们希望保留这个。
            #   所以只屏蔽 0 到 n_prompt-2，保留 n_prompt-1 位置（它预测 response 开头）。
            #
            # 类比：老师让学生续写"从前有座山，山里有座庙，"——
            #   学生不需要重复"从前有座山，山里有座庙，"这部分，
            #   只需要续写"庙里有个老和尚..."。prompt 部分的 loss 屏蔽就是这个意思。
            for i in range(n_prompt - 1):
                y[i] = -100

            # 收集处理后的序列
            input_ids.append(x)
            labels.append(y)

        # ==========================================
        # 步骤 5：统一填充到 block_size
        # ==========================================
        # 不同样本长度不同，但 Transformer 需要固定长度的输入矩阵。
        # pad_to 函数将每条序列截断/填充到 block_size 长度。

        def pad_to(ids, val):
            """将序列填充或截断到 block_size 长度。
            Args:
                ids: token ID 列表
                val: 填充值（input 用 2, label 用 -100）
            Returns:
                长度为 block_size 的列表
            """
            if len(ids) < self.block_size:
                # 序列过短：用 val 填充到 block_size。
                # 语法：[val] * N 生成包含 N 个 val 的列表。
                # ids + [val]*N 用 list 拼接，等价于 ids.extend([val, val, ...])
                ids = ids + [val] * (self.block_size - len(ids))
            # 语法：[:self.block_size] 确保最终长度不超过 block_size
            # （如果原序列就超过 block_size，上面 encode 时已经截断过了，这里再保底一次）
            return ids[:self.block_size]

        # ─── 转换为 PyTorch 张量 ───
        # 语法：列表推导式 for s in input_ids → 对每条样本调用 pad_to
        # torch.tensor(list_of_lists, dtype=...) 从嵌套列表构建 2D 张量。
        #
        # input 填充值为 2：
        #   为什么用 2？在 UTF-8 编码中，2 是"文本起始"(STX)控制字符。
        #   实际上任何非负整数都可以——模型只关心非填充位置的 token。
        #   选择 2（而非 0）是为了不与真实 token ID 混淆（0 可能是有效的词汇表条目）。
        #
        # label 填充值为 -100：
        #   -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index。
        #   填充位置的 label 设为 -100 意味着它们也不参与 loss 计算。
        #   这样模型既不看填充位置的内容（被 attention mask 忽略），
        #   也不对填充位置做预测（被 ignore_index 跳过），双重保险。
        #
        # dtype=torch.long → 64 位整数（int64），CrossEntropyLoss 要求 label 为 long 类型。
        x = torch.tensor([pad_to(s, 2) for s in input_ids], dtype=torch.long)
        y = torch.tensor([pad_to(s, -100) for s in labels], dtype=torch.long)

        # 返回 (输入张量, 目标张量)，形状均为 (batch_size, block_size)
        # xb 形状 (B, T)：模型输入 → 经过 Transformer → logits 形状 (B, T, vocab_size)
        # yb 形状 (B, T)：训练目标 → 与 logits 计算 CrossEntropyLoss
        #                 其中 prompt 区域为 -100，loss 自动忽略
        return x, y

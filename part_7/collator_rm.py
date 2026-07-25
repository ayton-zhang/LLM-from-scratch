# ==========================================
# 奖励模型成对数据整理器 (PairCollator)
# 职责：将偏好数据 (Prompt, Chosen, Rejected) 格式化为 SFT 风格文本，
#       通过 BPE/Byte 分词器编码，并截断与填充为成对的 Token 张量 (pos, neg)。
# ==========================================

from __future__ import annotations
from typing import List, Tuple
import torch

# Prefer BPE from Part 4, else ByteTokenizer from Part 3
# ─── 动态路径引入与多级依赖降级 ───
import sys
from pathlib import Path as _P

# 语法：_P(__file__).resolve().parents[1] 获取项目根目录 (llm_from_scratch)
# 尝试从 part_4 导入高性能 BPE 分词器 (BPETokenizer)
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False

# 若 BPE 分词器不可用，尝试从 part_3 导入兜底的字节分词器 (ByteTokenizer)
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

# 从 part_6 导入指令微调文本格式化工具 (Example, format_example)，复用对话模板规则
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_6'))
try:
    from formatters import Example, format_example  # reuse formatting
except Exception:
    pass

# ==========================================
# 成对数据整理器类
# ==========================================

class PairCollator:
    """Tokenize preference pairs into (pos, neg) input ids.
    We format as the SFT template with the 'chosen' or 'rejected' text as the Response.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None, vocab_size: int | None = None):
        # ─── 参数初始化 ───
        # block_size : 序列最大上下文长度（Prompt + Response），超越此长度会被截断，不足会被填充
        # bpe_dir    : 预训练 BPE 词表模型的保存目录路径
        # vocab_size : 词表大小（若未提供 BPE 模型，默认初始化 8000）
        self.block_size = block_size
        self.tok = None

        # 第一级尝试：使用 BPE 分词器
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size or 8000)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None

        # 第二级尝试：降级使用字节级分词器 (ByteTokenizer)
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()

        # 第三级：若均不可用则抛出异常，阻止程序静默运行
        if self.tok is None:
            raise RuntimeError("No tokenizer available.")

    @property
    def vocab_size(self) -> int:
        # 语法：@property 属性修饰器，将方法包装为只读属性 (可以通过 col.vocab_size 访问)
        # getattr 防御性编程：获取 self.tok.vocab_size，若对象无此属性则返回默认值 256
        return getattr(self.tok, 'vocab_size', 256)

    def _encode(self, text: str) -> List[int]:
        # ─── 统一编码接口封装 ───
        # 兼容不同分词器 API：支持返回 List[int] 或 torch.Tensor 的分词器，并提供 UTF-8 兜底
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            # 语法：isinstance 检查返回值是否为 PyTorch 张量，若是则转为 Python 原生列表
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        # 兜底方案：将字符串按 UTF-8 字节直接转为 0~255 的整数列表
        return list(text.encode('utf-8'))

    def collate(self, batch: List[Tuple[str, str, str]]):
        # ─── 核心批处理整理逻辑 ───
        # 输入输入数据：batch 为元组列表 [(prompt, chosen, rejected), ...]，包含 B 个成对样本
        pos_ids, neg_ids = [], []

        for prompt, chosen, rejected in batch:
            # 拼接并格式化：将 Prompt 与 Chosen/Rejected 按照对话模板 (SFT template) 组合为文本串
            pos_text = format_example(Example(prompt, chosen))
            neg_text = format_example(Example(prompt, rejected))

            # 分词编码并在线截断至 max_length (block_size)
            pos_ids.append(self._encode(pos_text)[:self.block_size])
            neg_ids.append(self._encode(neg_text)[:self.block_size])

        # 内部辅助函数：填充序列至固定长度 block_size
        # pad=2 代表填充 Token (Padding Token ID)
        def pad_to(x, pad=2):
            return x + [pad] * (self.block_size - len(x)) if len(x) < self.block_size else x[:self.block_size]

        # 语法：将填充后的二维整数列表转化为 PyTorch 64位长整型张量 (torch.long)
        # 数据变换：List[List[int]] → 形状为 (B, block_size) 的二维张量
        pos = torch.tensor([pad_to(x) for x in pos_ids], dtype=torch.long)
        neg = torch.tensor([pad_to(x) for x in neg_ids], dtype=torch.long)

        # 返回正样本与负样本成对张量，形状均为 (batch_size, block_size)
        return pos, neg
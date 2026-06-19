# ==========================================
# 组件：TextBPEBuffer —— 流式文本数据集
# ==========================================
# 把原始文本文件转为 PyTorch Dataset，供 DataLoader 批量产出 (x, y) 训练对。
#
# ==========================================
# 语言模型训练的数据构造方式
# ==========================================
# 核心思想：语言模型的任务是"给定前文，预测下一个 token"。
# 因此数据从同一个长序列上切出 x 和 y，y 就是 x 右移一位：
#
#   原始 token 序列：[tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7, ...]
#                    └── x ──┘
#                    │    └── y ──┘
#                         └── x ──┘
#                         │       └── y ──┘
#
#   每个训练样本位置 i 上：
#     x = ids[i    : i+block_size]    →  模型"看到"的上下文
#     y = ids[i+1  : i+block_size+1]  →  模型要"预测"的下一个 token
#
# 例如 block_size=3, i=0：
#   x = [tok0, tok1, tok2]   输入："我 喜欢 吃"
#   y = [tok1, tok2, tok3]   目标："喜欢 吃 苹果"
#   模型用 x 的每个位置预测 y 的对应位置：tok0→tok1, tok1→tok2, tok2→tok3
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
from tokenizer_bpe import BPETokenizer


class TextBPEBuffer(Dataset):
    """Memory-mapped-ish single-file dataset: tokenize once → long tensor of ids.
    get(idx) returns a (block_size,) slice; we construct (x,y) with shift inside collate.
    """

    # ==========================================
    # __init__：一次性把整个文件 tokenize 进内存
    # ==========================================
    # "Memory-mapped-ish" 的含义：
    #   严格的内存映射（memory-mapped file）是通过 mmap 系统调用，把磁盘文件
    #   直接映射到虚拟地址空间，OS 按需加载，不占物理内存。
    #   这里用的是一种"简化版"——一次性把整个文件读入内存，tokenize 成
    #   一个长 tensor。对于小型数据集（几百 MB 以下），这足够快且简单。
    #   对于更大的数据集（几十 GB），可以改造为真正的内存映射或分片加载。
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256):
        super().__init__()
        self.block_size = block_size

        # Path(path).read_text()：一次性读取文件全部文本到内存。
        # encoding='utf-8' 确保正确处理非 ASCII 字符（中文、emoji 等）。
        text = Path(path).read_text(encoding='utf-8')

        # tokenizer.encode(text)：把整段文本转为一个 Python list[int]。
        # torch.tensor(..., dtype=torch.long)：把 list 转为 1D 整数张量。
        # 此后 self.ids 是一个形状 (total_tokens,) 的张量，存储在内存中。
        # 例如 1MB 文本 → BPE 编码后约 500K token →
        #   500K × 8 bytes = 4MB 内存
        #   为什么是 8 bytes？torch.long = int64 = 64 位 = 8 字节。
        #   其实 token ID 只需 0~vocab_size 的范围，int16 或 int32 就够了，
        #   但 nn.Embedding 要求输入必须是 LongTensor（int64），这是 PyTorch 的硬性约定。
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # ==========================================
    # __len__：共有多少个训练样本
    # ==========================================
    def __len__(self):
        # 公式：len = total_tokens - block_size - 1
        #
        # 为什么 -block_size？
        #   最后一个训练样本 i 必须满足 i + block_size + 1 <= total_tokens，
        #   即 i <= total_tokens - block_size - 1。
        #   否则切片 ids[i+1 : i+block_size+1] 会越界。
        #
        # 为什么 -1？
        #   因为 y 比 x 多"偏移了一位"（y = ids[i+1 : i+block_size+1]），
        #   y 的最后一个元素是 ids[i+block_size]，比 x 的范围多 1。
        #   所以额外少了一个位置。
        #
        # 举例：total_tokens=100, block_size=8
        #   i=0:  x=ids[0:8],  y=ids[1:9]    ✓
        #   i=90: x=ids[90:98], y=ids[91:99]  ✓
        #   i=91: x=ids[91:99], y=ids[92:100] ✓ (最后一个有效样本)
        #   i=92: y=ids[93:101] → 越界！
        #   len = 100-8-1 = 91 ✓
        #
        # max(0, ...)：如果文本太短（total_tokens <= block_size+1），
        # 返回 0 个样本（而非负数），避免 DataLoader 崩溃。
        return max(0, self.ids.numel() - self.block_size - 1)

    # ==========================================
    # __getitem__：取第 i 个训练样本的 (x, y)
    # ==========================================
    # 这是 PyTorch Dataset 协议的核心方法——DataLoader 的每个 worker 会调用
    # dataset[i] 来取第 i 个样本，然后 batch 在一起。
    def __getitem__(self, i: int):
        # x 和 y 都是从同一个长序列上取的长度为 block_size 的切片：
        #   x = ids[i    : i+block_size]    ← 模型看到的 token 序列
        #   y = ids[i+1  : i+block_size+1]  ← 模型要预测的下一个 token
        #
        # 两个切片的长度都是 block_size，但它们"错位"了 1 个位置。
        # 因为每个 token 位置都要预测"紧跟在它自己后面的那个 token"。
        #
        # 为什么这里返回 (x, y) 而不是在 collate_fn 中构造 y？
        #   直接在 __getitem__ 里返回 (x,y) 更直观，DataLoader 默认的 collate
        #   会把它们分别 stack 成 (B, block_size) 和 (B, block_size)。
        #
        # 语法：ids[i:i+N] 是 Python 切片，对 tensor 同样适用。
        # 注意不是 ids[i][i+block_size]，那是二维索引（先取第 i 个标量，再切片——报错）。
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y


# ==========================================
# make_loader：创建 DataLoader 的工厂函数
# ==========================================
def make_loader(path: str, tokenizer: BPETokenizer, block_size: int, batch_size: int, shuffle=True) -> DataLoader:
    """一次性工厂函数：创建 Dataset + 包装为 DataLoader。

    参数说明：
      path       : 训练文本文件的路径（.txt）
      tokenizer  : 已训练/加载的 BPE 分词器（或 None = 字节级回退）
      block_size : 每个训练样本的 token 数量
      batch_size : 每个 batch 有多少个样本
      shuffle    : 是否随机打乱样本顺序（训练时 True，验证时 False）

    返回值：torch.utils.data.DataLoader，迭代产出 (xb, yb) 元组，
            xb 形状 (B, block_size)，yb 形状 (B, block_size)。
    """
    ds = TextBPEBuffer(path, tokenizer, block_size)

    # DataLoader 的参数：
    #   ds：实现了 __len__ 和 __getitem__ 的 Dataset 对象。
    #   batch_size：每次取多少个样本 stack 在一起。
    #   shuffle=True：每 epoch 随机打乱样本顺序，防止模型记住数据顺序。
    #     PyTorch 内部用 RandomSampler 实现——不打乱的话相邻 batch 的文本高度相关，
    #     梯度估计会有偏（batch 之间的独立性低）。
    #   drop_last=True：丢弃最后一个不足 batch_size 的"零头"batch。
    #     为什么？batch norm（如果用了的话）需要 batch 大小一致；
    #     更重要的：最后一个 batch 太小时（如只剩 2 个样本，正常的 1/16），
    #     梯度估计不稳定，可能把训练搞偏。设 True 宁可少用几个样本。
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

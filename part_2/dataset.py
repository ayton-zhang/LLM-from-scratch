# 这一行是 Python 的特殊用法，允许在定义类的时候，在类型提示里提前使用一些还没加载完的特性。
from __future__ import annotations

# pathlib 是 Python 内置的处理文件路径的工具库，比传统的 os.path 更好用。
from pathlib import Path

# 导入 PyTorch 库，这是深度学习最常用的兵器库。
import torch

# 定义一个名为 ByteDataset 的类（就像制造一个叫“字节数据集”的模具）
class ByteDataset:
    """保存文本文件的原始字节，并为语言模型(LM)生成 (x, y) 数据块。
    - block_size: 序列长度（也就是模型的上下文窗口大小，比如一次看 256 个字符）
    - split: 用于训练的数据比例（剩下的用来做验证/测试）
    """
    
    # __init__ 是初始化方法（构造函数）。当你创建一个 ByteDataset 对象时，这段代码首先运行。
    # path: str 这种写法叫“类型提示”，意思是 path 这个参数应该是个字符串。
    # block_size: int = 256 意思是它应该是个整数，如果不传这个参数，默认就是 256。
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        
        # Path(path) 把字符串变成路径对象，.read_bytes() 直接把整个文件作为“原始字节”（0-255的数字）读进内存。
        data = Path(path).read_bytes()
        
        # list(data) 把字节序列变成普通的 Python 列表。
        # torch.tensor(...) 把这个普通列表转换成 PyTorch 的“张量”（一种能在显卡上高效运算的超级数组）。
        # dtype=torch.long 指定这些数字的数据类型是 64 位整数。
        data = torch.tensor(list(data), dtype=torch.long)
        
        # len(data) 获取数据的总长度。乘以 split (0.9) 后，算出前 90% 数据所在的位置（索引 n）。
        # int(...) 确保算出来的位置是个整数，不能把一个字节劈成两半。
        n = int(len(data) * split)
        
        # [ : n] 是 Python 的切片语法，意思是“从开头一直取到第 n 个”。这里把前 90% 存为训练集。
        self.train = data[:n]
        
        # [n : ] 意思是“从第 n 个一直取到最后”。这里把后 10% 存为验证集。
        self.val = data[n:]
        
        # 把传进来的 block_size 保存到这个对象自己身上，方便以后使用。
        self.block_size = block_size

    # 这个方法用来随机获取一批（batch）训练数据。
    # which 代表要取哪个数据集（'train' 或 'val'），batch_size 是一次取几道题，device 是指放在 CPU 还是 GPU 上。
    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val
        
        # 1. +1的原因在于y是x的右移版本，如果不加1，y的最后一个token就会超出buf的范围。
        # 2. 是>的原因是下方ix计算randint不能为[0, 0].
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        
        # 核心逻辑 1：生成随机数。
        # torch.randint(起点, 终点, 形状)。这里在 [0 到 总长度减去窗口大小] 之间，随机挑出 batch_size 个起始位置。
        # 比如 batch_size 是 4，ix 可能就是 [15, 1024, 56, 9999]。
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        
        # 核心逻辑 2：生成题目 x（输入数据）。
        # 这里用到了 Python 的“列表推导式” [ ... for i in ix ]，意思是遍历刚才选出的随机起始点 i。
        # buf[i : i+self.block_size] 意思是截取从 i 开始，长度为 block_size 的一截数据。
        # torch.stack(...) 把截出来的这一堆小片段，像叠千层饼一样堆叠成一个新的二维张量。
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        
        # 核心逻辑 3：生成答案 y（目标数据）。
        # 注意看索引！它是从 i+1 开始，到 i+1+block_size 结束。也就是整体比 x 往后错了一位。
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        
        # .to(device) 会把生成好的 x 和 y 搬运到指定的设备上（比如搬到显卡 GPU 上进行加速运算），然后返回。
        return x.to(device), y.to(device)
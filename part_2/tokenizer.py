from __future__ import annotations  # 导入未来版本的注解语法，使类型提示更灵活，如使用 | 代替 Union
import torch  # 导入PyTorch库，用于张量操作

class ByteTokenizer:  # 定义一个字节级分词器类，继承自object（默认）
    """Ultra-simple byte-level tokenizer.  # 类文档字符串，描述类的功能
    - encode(str) -> LongTensor [N]  # encode方法：输入字符串，返回长整型张量，形状为[N]（N是字节数）
    - decode(Tensor[int]) -> str  # decode方法：输入整型张量，返回字符串
    - vocab_size = 256  # 词汇表大小为256（字节级，0-255）
    """
    def encode(self, s: str) -> torch.Tensor:  
        # 定义encode方法，输入参数s为字符串，返回torch.Tensor
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)  
        # 将字符串s编码为UTF-8字节列表，每个字符都可以被一个或者多个字节表示，且范围为0-255

    def decode(self, ids) -> str:  
        # 定义decode方法，输入参数ids为张量或列表，返回字符串
        if isinstance(ids, torch.Tensor):  # 检查ids是否为torch.Tensor类型
            ids = ids.tolist()  # 如果是张量，转换为Python列表
        return bytes(ids).decode('utf-8', errors='ignore')  
    # 将列表转换为bytes对象，然后解码为UTF-8字符串，忽略解码错误

    @property  # 装饰器，表示这是一个属性方法，不是普通方法
    def vocab_size(self) -> int:  # 用于构建embedding嵌入层，其形状为[vocab_size, embedding_dim]
        return 256  # 返回词汇表大小256（字节级分词器的固定大小）
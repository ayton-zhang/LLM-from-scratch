"""1.1 Positional encodings (absolute learned + sinusoidal)."""  # 两种位置编码：学习型和正弦型
import math  # 数学库，用来计算对数
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块

class LearnedPositionalEncoding(nn.Module):
    """学习的位置编码：用可训练的嵌入层来自动学习每个位置的编码向量"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()  
        # 初始化基类
        self.emb = nn.Embedding(max_len, d_model)  
        # 创建嵌入层：把（0到max_len-1）的位置编号映射到 d_model 维向量

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model) — 我们只需要知道序列长度 T 和设备信息 device
        B, T, _ = x.shape  
        # 解包形状：B 批量，T 序列长度，_ 忽略第三维
        pos = torch.arange(T, device=x.device)  
        # 创建位置索引 [0,1,2,...,T-1]，放在 x 的同一设备上
        pos_emb = self.emb(pos)  
        # 用嵌入层查表，把位置索引转换成位置向量，(T, d_model)
        return x + pos_emb.unsqueeze(0)  
        # unsqueeze(0) 在开头加批维度变成 (1,T,d_model)，广播加到 x 上

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码：用正弦余弦函数生成固定的位置编码，不需要训练"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()  # 初始化基类
        # 创建全零矩阵，形状 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  
        # 位置 [0,1,2,...,max_len-1]，变成列向量 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        # 计算分母中的缩放因子：
        # exp(-2i/d_model * log(10000)) = (1/10000)^(2i/d_model)，
        # 这样不同维度频率不同
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        # 偶数列用正弦：每个位置的偶数维度 = sin(position/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  
        # 奇数列用余弦：每个位置的奇数维度 = cos(position/10000^(2i/d_model))
        self.register_buffer('pe', pe)  
        # 把位置编码 pe 注册为缓冲区，随模型一起保存但不参与训练 (max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape  
        # 解包形状：B 批量，T 序列长度，_ 忽略第三维
        return x + self.pe[:T].unsqueeze(0)  
        # 从缓冲区取前 T 个位置编码，加上批维度 (1,T,d_model)，广播加到 x 上
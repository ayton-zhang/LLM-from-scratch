import math  # 导入数学库，主要为了开根号
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络层模块，别名 nn
import torch.nn.functional as F  # 常用函数库，别名 F
from attn_mask import causal_mask  # 从 attn_mask.py 里拿因果掩码函数

class MultiHeadSelfAttention(nn.Module):
    """1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
      x:      (B, T, d_model)
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      ctx:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()  # 初始化基类，必须的
        assert d_model % n_head == 0, "d_model must be divisible by n_head"  # 确保每个头维度是整数
        self.n_head = n_head  # 头数
        self.d_head = d_model // n_head  # 每个头的维度
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # 一个线性层算 QKV 叠一起
        self.proj = nn.Linear(d_model, d_model, bias=False)  # 输出投影层
        self.dropout = nn.Dropout(dropout)  # dropout 层
        self.trace_shapes = trace_shapes  # 是否打印形状（调试用）

    def forward(self, x: torch.Tensor):  # x 形状 (B,T,d_model)
        B, T, C = x.shape  # B 批量，T 序列长度，C 模型维度
        qkv = self.qkv(x)                          # 一次性给出 qkv，(B,T,3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)  # 拆成 3 份，(B,T,3,heads,dim)
        if self.trace_shapes:
            print("qkv view:", qkv.shape)  # 输出 qkv 的形状
        q, k, v = qkv.unbind(dim=2)               # 拆成 q,k,v，(B,T,heads,dim)
        q = q.transpose(1, 2)                      # 合理换维度，(B,heads,T,dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        scale = 1.0 / math.sqrt(self.d_head)  # 缩放因子，防止点积太大
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # 计算注意力分数，(B,heads,T,T)
        mask = causal_mask(T, device=x.device)  # causal 掩码，上三角置 True
        attn = attn.masked_fill(mask, float('-inf'))  # 未来位置填 -inf
        w = F.softmax(attn, dim=-1)  # softmax 概率
        w = self.dropout(w)  # 随机 dropout
        ctx = torch.matmul(w, v)                  # 权重和 v 相乘，(B,heads,T,dim)
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # 合并头，恢复 (B,T,d_model)
        out = self.proj(out)  # 线性变换得到最终输出
        if self.trace_shapes:
            print("out:", out.shape)
        return out, w  # 返回输出和注意力权重

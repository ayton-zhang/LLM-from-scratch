# ==========================================
# 奖励模型 (Reward Model, RM) 架构实现
# 职责：基于双向 Transformer 编码器 (Encoder)，对输入文本序列 (Prompt + Response) 进行特征抽取，
#       通过掩码均值池化 (Masked Mean Pooling) 消除 Padding Token 的影响，最后使用线性层映射为标量偏好得分 r ∈ ℝ。
# ==========================================

from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """Transformer encoder → pooled representation → scalar reward.
    Bidirectional encoder is fine for reward modeling (not used for generation).
    """
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1):
        # ─── 参数初始化 ───
        # vocab_size : 词表大小，决定 Token 嵌入矩阵的行数
        # block_size : 上下文最大序列长度，决定位置嵌入矩阵的行数
        # n_layer    : Transformer 编码器的堆叠层数 (默认 4 层)
        # n_head     : 多头自注意力机制的头数 (默认 4 头)
        # n_embd     : 词嵌入向量维度 / 隐层向量维度 (默认 256)
        # dropout    : Dropout 随机失活概率，防止过拟合 (默认 0.1)
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        # 1. 词嵌入层：将 Token ID 映射为 n_embd 维连续特征向量
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # 2. 绝对位置嵌入层：为序列中 0 ~ block_size-1 位置学习可训练的位置编码
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # 3. 单层 Transformer 编码器构建：
        #   d_model          : 隐层维度
        #   nhead            : 多头注意力头数
        #   dim_feedforward  : FFN 隐藏层维度，按照惯例设为 4 * n_embd
        #   activation       : 激活函数，选择 GELU 激活函数
        #   batch_first=True : 规定输入张量的维度顺序为 (Batch, SeqLen, Embedding)
        enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
                                               dropout=dropout, activation='gelu', batch_first=True)
        # 4. 堆叠 n_layer 层 Transformer 编码器层
        # 注意：RM 仅用于给完整文本打分，不需要因果下三角掩码 (Causal Mask)，双向注意力能提取更丰富的上下文特征
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        # 5. 层归一化与标量投影头
        self.ln = nn.LayerNorm(n_embd)
        # 标量输出头：将 n_embd 维度的句向量映射为 1 维标量奖励分 (Scalar Reward Score)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor):
        # ─── 前向传播与掩码均值池化 ───
        # 语法：B, T = x.shape 是元组解包。输入 x 的形状为 (Batch_Size, Seq_Len)
        B, T = x.shape

        # 构造位置 ID 向量：形状为 (T,) → unsqueeze(0) 拓展为 (1, T)，以便在 Batch 维度广播
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        # 词嵌入 + 位置嵌入：
        # tok_emb(x) 形状: (B, T, n_embd)
        # pos_emb(pos) 形状: (1, T, n_embd)
        # 广播相加后 h 形状: (B, T, n_embd)
        h = self.tok_emb(x) + self.pos_emb(pos)

        # 生成 Padding 填充位置掩码：
        # 约定 ID=2 为 Padding Token。x == 2 返回布尔张量 (B, T)，Pad 位置为 True，非 Pad 位置为 False
        pad_mask = (x == 2)

        # 传入 Transformer 编码器进行多层双向自注意力编码：
        # src_key_padding_mask 会将 pad_mask 中 True 的位置在计算 Attention 权重时屏蔽
        # h 变换前形状: (B, T, n_embd) → 变换后形状: (B, T, n_embd)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)

        # ─── 关键步骤：掩码均值池化 (Masked Mean Pooling) ───
        # 目的：将序列中非 Pad 位置的 Token 向量求平均，提取句级别的特征向量
        # ~pad_mask 将布尔值取反（真实 Token 为 True，Pad 为 False）
        # .float().unsqueeze(-1) 将 True/False 转换为 1.0/0.0 并扩维，mask 形状为 (B, T, 1)
        mask = (~pad_mask).float().unsqueeze(-1)

        # 广播相乘：(B, T, n_embd) * (B, T, 1)，将所有 Pad 位置的向量清零
        # .sum(dim=1)：在时间维度 T 上求和，h_sum 形状为 (B, n_embd)
        h_sum = (h * mask).sum(dim=1)

        # 计算每个样本中真实有效 Token 的个数：
        # mask.sum(dim=1) 形状为 (B, 1)；.clamp_min(1.0) 保证分母最小为 1.0，防止全为 Pad 时除以 0 报错 NaN
        len_ = mask.sum(dim=1).clamp_min(1.0)

        # 向量除以有效长度得到平均句向量，pooled 形状: (B, n_embd)
        pooled = h_sum / len_

        # 经过线性头映射为标量得分：(B, n_embd) → (B, 1)
        # squeeze(-1) 移除最后一维，最终输出 r 形状: (B,)，即当前 Batch 中每条样本的奖励得分
        r = self.head(pooled).squeeze(-1)  # (B,)
        return r
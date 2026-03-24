from __future__ import annotations
import math
import torch
import torch.nn as nn          # nn 是 Neural Network (神经网络) 的缩写，包含了各种现成的网络层
import torch.nn.functional as F # F 包含了一些不需要权重的纯数学函数运算（比如加减乘除的高级版）

# ==========================================
# 组件一：自注意力机制 (模型的“雷达”)
# ==========================================
class CausalSelfAttention(nn.Module): # 继承 nn.Module，说明这是神经网络的一个零件
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__() # 拜见祖师爷，初始化父类
        
        # assert 确保总特征数(n_embd)能被多头数量(n_head)整除，不然没法平均分配
        assert n_embd % n_head == 0 
        
        self.n_head = n_head # 头的数量 (相当于把雷达分成几个不同方向的探测器)
        self.d_head = n_embd // n_head # 每个头分到的维度大小
        
        # 核心：Q(查询), K(钥匙), V(价值)。
        # 这里用一个全连接层(Linear)把输入特征直接放大3倍，一次性把 Q, K, V 都算出来，比分开算更快。
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # 算完之后，再用一个 Linear 把结果压缩回原来的维度
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # Dropout 是一种防止模型“死记硬背”(过拟合)的机制，它会随机把一些神经元打晕(设为0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # forward 是 PyTorch 的核心，定义了数据怎么流过这个零件
        # B = Batch Size (批次大小，也就是一次处理几道题)
        # T = Time/Sequence Length (句子长度，比如一次处理 256 个字)
        # C = Channels/Embedding Dim (每个字用多少个数字表示，即 n_embd)
        B, T, C = x.shape 
        
        # .view() 相当于“变形金刚”。把算出来的 3*C 切分成 Q,K,V 三份，并分给不同的头(n_head)。
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        
        # .unbind(dim=2) 把刚才的 3 份沿着维度拆开，分别赋值给 q, k, v
        q, k, v = qkv.unbind(dim=2)
        
        # .transpose(1, 2) 是维度对调。为了让底层计算更快，要把 T(句子长度) 和 头数 换个位置。
        # 现在的形状变成了 (B, n_head, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 这是一个缩放因子（作者算出来了但其实没用上，因为下面的函数内部自带了缩放功能）
        scale = 1.0 / math.sqrt(self.d_head) 
        
        # 魔法函数！这是 PyTorch 官方提供的超级加速版注意力计算公式（FlashAttention）。
        # is_causal=True 最关键：它蒙住模型的眼睛，让模型只能看到前面的字，绝对不能偷看后面的答案！
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        
        # 算完之后，把刚才对调的维度再换回来。.contiguous() 是在内存里把数据排列整齐，方便再用 .view() 变回 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 最后经过一个线性层输出
        y = self.proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ---- Tiny GPT ----
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

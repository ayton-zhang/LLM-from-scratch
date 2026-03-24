from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 组件一：自注意力机制
# ==========================================
class CausalSelfAttention(nn.Module): # 继承 nn.Module，说明这是神经网络的一个零件
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__() # 拜见祖师爷，初始化父类
        
        # assert 确保总特征数(n_embd)能被多头数量(n_head)整除，不然没法平均分配
        assert n_embd % n_head == 0 
        
        self.n_head = n_head # 头的数量
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

# ==========================================
# 组件二：前馈神经网络 (模型的“思考区”)
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # nn.Sequential 就是一条流水线，数据进去会依次经过括号里的每一道工序：
        self.net = nn.Sequential(
            # 第一步：把特征维度放大 mult 倍（通常是 4 倍），让模型在更高维度思考
            nn.Linear(n_embd, mult * n_embd),
            # 第二步：GELU 激活函数。你可以把它当成模型脑细胞的“兴奋剂”，加入非线性，让它能学复杂的逻辑
            nn.GELU(),
            # 第三步：思考完了，再把维度压缩回原来的大小
            nn.Linear(mult * n_embd, n_embd),
            # 第四步：还是随机打晕一些神经元防过拟合
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 组件三：基础模块 (一层大脑皮层)
# ==========================================
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        # LayerNorm 是归一化。因为模型里数字乘来乘去容易爆炸(极大)或消失(极小)，LayerNorm 负责把数字强行按回正常范围。
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        # 重点语法：x = x + ... 叫做“残差连接”。
        # 意思是：我不光要加上刚才“雷达(attn)”算出的新信息，还要保留我原本自己(x)的信息。这能让极深的网络更好训练。
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

# ==========================================
# 组件四：完整的微型 GPT 模型
# ==========================================
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size # 最大上下文长度
        
        # 两个字典：
        # tok_emb: 词嵌入字典。把词的ID（比如 102）变成一堆带有含义的数字(向量)。
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # pos_emb: 位置嵌入字典。告诉模型这个词在句子里的位置（第一还是第二），因为 Transformer 默认是路痴，不知道词的顺序。
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.drop = nn.Dropout(dropout)
        
        # nn.ModuleList 是个装神经网络零件的列表。这里用一个 for 循环，一口气造了 n_layer 层我们刚才定义的 Block。
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd) # 最后一层收尾的归一化
        self.head = nn.Linear(n_embd, vocab_size, bias=False) # 归一化之后的输出层（最后负责投票的裁判，决定下一个词输出哪个）

        # 遍历上面的所有零件，给它们初始化随机权重
        self.apply(self._init_weights)

    # 这是一个辅助函数，用来给各种权重赋初始值的。标准做法是给一个均值为0，标准差为0.02的正态分布随机数。
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # 模型的前向传播（训练和推理的核心路径）
    # targets: 可选参数。如果有答案传进来，就顺便把分数(loss)算一下；如果不传答案，只管输出预测(logits)。
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape # B是批次，T是当前句子的长度
        assert T <= self.block_size # 不能超过模型允许的最大阅读长度
        
        # 生成一个从 0 到 T-1 的位置序列：[0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        
        # 核心：当前输入的特征 = 词汇的特征 + 它的位置特征
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        
        # 把数据推入那好几层的 Block 中，一层层往上思考加工
        for blk in self.blocks:
            x = blk(x)
            
        x = self.ln_f(x)
        logits = self.head(x) 
        # logits的形状是 (B, T, vocab_size)，每个位置都有一个词汇表大小的得分
        
        loss = None
        if targets is not None:
            # cross_entropy 要求输入时二维的
            # 所以把 logits 从 (B, T, vocab_size) 变成 (B*T, vocab_size[每个词的得分])
            # targets 从 (B, T) 变成 (B*T)，每个位置对应一个正确的词ID
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    # ==========================================
    # 推理生成函数
    # ==========================================
    @torch.no_grad() # 魔法指令：告诉系统“我现在是实战考试，不是在学习(训练)”。这会省下巨量内存，因为不需要记住解题思路(梯度)了。
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int | None = 50, top_p: float | None = None):
        from utils import top_k_top_p_filtering
        self.eval()
        
        # 如果你没给模型提示词，它就自己放一个换行符(编号10)作为起点
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
            
        # 循环：打算让模型写多少个新词，就循环多少次
        for _ in range(max_new_tokens):
            # 如果句子太长了，只截取最后 block_size 个字给模型看（因为它的脑容量只有这么大）
            idx_cond = idx[:, -self.block_size:]
            
            # 把这段文字扔进模型，拿到原始打分 logits
            logits, _ = self(idx_cond)
            
            # 我们只要最后一个字(刚才新生成的字)的预测分数。/ temperature 是控制发散程度。
            # temperature 越小（比如 0.1），模型越死板保守；越大（比如 1.5），模型越放飞自我瞎编。
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # 过滤掉那些得分极低的“胡言乱语”，只留得分最高的 K 个候选人
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            
            # 把原始得分转换成加起来等于 100% 的概率 (Softmax)
            probs = torch.softmax(logits, dim=-1)
            
            # multinomial：按概率摇号抽奖！得分高的词抽中的概率大，但不绝对。抽出 1 个新词。
            next_id = torch.multinomial(probs, num_samples=1)
            
            # 把新抽出来的一个词，拼接到原来的句子里。进入下一次循环！
            idx = torch.cat([idx, next_id], dim=1)
            
        return idx # 循环结束，返回写好的大长篇
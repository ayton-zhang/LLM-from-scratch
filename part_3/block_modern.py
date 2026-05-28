# ==========================================
# 组件：TransformerBlockModern —— 现代 Transformer 解码器块
# ==========================================
# 与 Part 2 的 TransformerBlock 相比，本模块做了三处关键升级：
#   1. 用 RMSNorm  替换 LayerNorm  （去掉均值归一化，速度更快，效果相当）
#   2. 用 SwiGLU   替换 GELU FFN  （门控激活函数，更强的非线性表达，LLaMA/GPT-4 采用）
#   3. 用支持 RoPE 的注意力层替换原始 CausalSelfAttention（位置编码更灵活，支持 KV Cache）
import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern

class TransformerBlockModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        # 语法：`A if 条件 else B` 是 Python 三元表达式。
        # use_rmsnorm=True 时选用 RMSNorm（只做方差归一化，无均值偏移，更快）；
        # 否则退回到标准 LayerNorm（与 Part 2 兼容）。
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        # 注意力子层前的归一化（Pre-Norm 结构）：
        # 这里是pre-norm结构：先归一化再做注意力，训练更稳定，梯度不容易爆炸/消失。
        self.ln1 = Norm(n_embd)

        # 现代因果自注意力层，支持 RoPE、滑动窗口、注意力水槽、GQA 等高级特性。
        # 参数说明：
        #   n_embd        : 隐状态维度（词向量维度）
        #   n_head        : Query 头数
        #   dropout       : 注意力权重的随机丢弃比例，防止过拟合
        #   rope          : 是否启用旋转位置编码（True = 现代做法）
        #   max_pos       : RoPE 预计算的最大序列长度
        #   sliding_window: 局部注意力窗口大小，None = 全局注意力
        #   attention_sink: 强制保留开头的 K 个 token，防止极长序列"遗忘"开头
        #   n_kv_head     : KV 头数（GQA/MQA），None = 与 n_head 相同（标准 MHA）
        self.attn = CausalSelfAttentionModern(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)

        # FFN 子层前的归一化（同样是 Pre-Norm）
        self.ln2 = Norm(n_embd)

        # 前馈网络（FFN）：
        # use_swiglu=True 时使用 SwiGLU 门控 FFN（Swish 激活 × 线性门）
        # 否则退回到经典的 Linear → GELU → Linear 结构（与 Part 2 兼容）
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, kv_cache=None, start_pos: int = 0):
        # ─── 第一子层：带残差连接的因果自注意力 ───
        # 先对 x 做归一化（ln1），再送入注意力层。
        # 语法：`a, kv_cache = self.attn(...)` 是元组解包；
        #   attn 返回 (注意力输出张量, 更新后的KV缓存)，分别赋给 a 和 kv_cache。
        # kv_cache  : 本层的 KV 缓存对象，推理时传入已缓存的历史 K/V，避免重复计算。
        # start_pos : 当前这批 token 在完整序列中的起始下标，用于 RoPE 计算正确的旋转角度。
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)

        # 残差连接：x = x + 注意力输出。
        # 残差路径让梯度可以"抄近道"直接回传，极大缓解深层网络的梯度消失问题。
        x = x + a

        # ─── 第二子层：带残差连接的前馈网络 ───
        # 同样先归一化（ln2），再过 FFN，最后加回残差。
        # FFN 负责在每个位置做"逐 token 的特征变换"，注意力负责"跨 token 的信息混合"，
        # 两者分工合作，共同提升模型表达能力。
        x = x + self.ffn(self.ln2(x))

        # 返回更新后的隐状态和本层 KV 缓存（供下一个生成步骤复用）
        return x, kv_cache
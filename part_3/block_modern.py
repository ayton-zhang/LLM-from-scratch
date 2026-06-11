# ==========================================
# 组件：TransformerBlockModern —— 现代 Transformer Block
# ==========================================
# 这是 Transformer 的"积木块"——模型里堆叠 n_layer 个本 Block 就构成了完整模型。
#
# 与 Part 2 的 TransformerBlock 相比，本 Block 做了三处升级：
#   1. Pre-Norm 架构：归一化放在注意力/FFN 之前（而非之后），训练更稳定
#   2. 条件组件选择：通过 use_rmsnorm / use_swiglu 开关灵活切换新旧实现
#   3. 传递 KV Cache：forward 返回更新后的缓存对象，支持自回归生成
#
# 每个 Block 内部结构（Pre-Norm 风格）：
#   x → ln1 → attn → +x (残差) → ln2 → ffn → +x (残差) → 输出
#
# 直觉理解 Pre-Norm vs Post-Norm：
#   原始论文用的是 Post-Norm（attn 完了再归一化），
#   现代模型普遍用 Pre-Norm（先归一化再 attn/ffn），因为：
#     - 归一化后再做 attn/ffn，输入分布稳定，梯度流动更顺畅
#     - 训练初期不容易出现 loss 不降（Nan/Inf）的情况
#     - 可以用更大的学习率训练
# 类比：
#   Post-Norm = 考完试再对答案（事后再修正，容易积累偏差）
#   Pre-Norm  = 考之前先复习重点（先归一化到同一基准，再发挥）
import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern

class TransformerBlockModern(nn.Module):
    # ==========================================
    # Block 组件初始化
    # ==========================================
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        # 参数说明（与 GPTModern 保持一致，通过模型一层层传递下来）：
        #   n_embd         : 隐藏层维度
        #   n_head         : 注意力头数
        #   dropout        : 随机丢弃率
        #   use_rmsnorm    : True → RMSNorm（更快）；False → LayerNorm
        #   use_swiglu     : True → SwiGLU FFN（非线性更强）；False → 普通 GELU MLP
        #   rope           : True → RoPE 位置编码；False → 无位置信息
        #   max_pos        : RoPE 预计算的最大序列长度
        #   sliding_window : 局部注意力窗口，None = 全局注意力
        #   attention_sink : 注意力水槽大小
        #   n_kv_head      : KV 头数（GQA），None = 标准 MHA
        super().__init__()

        # ─── 条件选择归一化层 ───
        # 语法：`Norm = RMSNorm if use_rmsnorm else nn.LayerNorm` 是类引用赋值（非实例化）。
        # 这里把"归一化类"存到变量 Norm 中，下面两行都用同一个类创建实例。
        # 这比写两个 if-else 更简洁，而且确保 ln1 和 ln2 使用同一种归一化类型。
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        # ln1：注意力之前的前置归一化。
        # 注意：RMSNorm 的参数只有 n_embd（隐藏维度），不像 LayerNorm 还需要 eps 等额外参数。
        # RMSNorm 默认 eps=1e-5 已内置在 RMSNorm 类里，这里不需要显式传入。
        self.ln1 = Norm(n_embd)

        # 因果自注意力层：包含了 QKV 投影、RoPE 旋转、KV Cache 拼接、
        # 滑动窗口裁剪、GQA 扩展、缩放点积注意力、多头拼接的完整注意力实现。
        self.attn = CausalSelfAttentionModern(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)

        # ln2：FFN 之前的前置归一化。
        self.ln2 = Norm(n_embd)

        # ─── 条件选择前馈网络 ───
        # 两种备选方案，由 use_swiglu 开关决定：
        #
        # SwiGLU（use_swiglu=True）：门控前馈网络。
        #   内部结构：两个线性投影（gate + value）→ SiLU 激活 → 逐元素相乘 → 输出投影。
        #   mult=4 表示中间维度 = 4 * n_embd（LLaMA 标配，其实是把两个 4x 投影算进总的参数量）。
        #   效果：非线性更强，建模能力更好，但参数量是普通 MLP 的约 1.5 倍。
        #
        # 普通 MLP（use_swiglu=False）：经典两层前馈。
        #   nn.Sequential 把多层按顺序串成一个模块：
        #     Linear(n_embd → 4*n_embd) → GELU 激活 → Linear(4*n_embd → n_embd) → Dropout
        #   GELU 是 Gaussian Error Linear Unit，比 ReLU 更平滑（处处可导），
        #   但比 SwiGLU 的非线性表达能力差。
        #
        # 语法：`A if 条件 else B` 三元表达式，等价于：
        #   if use_swiglu:
        #       self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout)
        #   else:
        #       self.ffn = nn.Sequential(nn.Linear(...), nn.GELU(), nn.Linear(...), nn.Dropout(...))
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )

    # ==========================================
    # forward：Pre-Norm + 注意力 + 残差 + FFN
    # ==========================================
    def forward(self, x, kv_cache=None, start_pos: int = 0):
        # 参数说明：
        #   x        : 输入隐状态，形状 (B, T, C)
        #   kv_cache : 本层的 KV Cache 对象（或 None）
        #   start_pos: 当前这批 token 在完整序列中的起始位置（传给 RoPE）
        # 返回值：
        #   x        : 输出隐状态，形状 (B, T, C)，与输入形状相同
        #   kv_cache : 更新后的 KV Cache 对象

        # ─── 注意力子层（Pre-Norm + 残差连接）───
        # 步骤拆解：
        #   1. self.ln1(x)：先对输入做归一化，确保 Q/K/V 投影的输入分布稳定。
        #      这是 Pre-Norm 的核心——在变换前把数据"摆正"。
        #   2. self.attn(...)：计算因果自注意力（内部包含 QKV 投影、RoPE、缓存拼接、SDPA）。
        #      返回 (注意力输出 a, 更新后的 KV Cache)。
        #   3. x = x + a：残差连接——把注意力输出"加回"原始输入。
        #
        # 语法：`a, kv_cache = self.attn(...)` 是元组解包，
        # attn.forward() 返回 (注意力输出张量, 更新后的缓存对象)。
        #
        # 为什么需要残差连接？
        #   深层网络在反向传播时梯度很容易消失（每一层都乘一个小数，N 层后趋近于 0）。
        #   残差连接提供了一条"高速公路"（identity shortcut），梯度可以直接穿过加法节点
        #   回流到前面的层，使深层 Transformer 的训练成为可能。
        #   直觉类比：高速公路和乡间小路并行——你可以走乡间小路（经过注意力/FFN 变换），
        #   也可以直接走高速公路（残差），两路汇合后继续前进。
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a

        # ─── FFN 子层（Pre-Norm + 残差连接）───
        # 步骤拆解：
        #   1. self.ln2(x)：先归一化（注意这里是"注意力之后"的 x，已包含残差结果）。
        #   2. self.ffn(...)：通过前馈网络做非线性变换。
        #      SwiGLU 内部会做升维 → 门控激活 → 降维的过程，
        #      可以理解为"把 token 的信息重新组织、提炼一遍"。
        #   3. x = x + ...：残差连接，把 FFN 输出加回输入。
        #
        # 为什么要先归一化再 FFN？为什么归一化放在这里而不是之前？
        #   所有现代 Transformer（GPT-2 起）都使用 Pre-Norm：归一化放在
        #   注意力/FFN 之前。这样每层的输入分布基本一致，训练更稳定。
        #   Post-Norm（归一化放在加法之后）在深层网络中容易出现梯度问题。
        x = x + self.ffn(self.ln2(x))

        # 返回更新后的隐状态和缓存：
        #   x 形状与输入相同 (B, T, C)，但每个 token 的表示已被注意力+FFN 更新。
        #   kv_cache 传给 model_modern.py 的 forward，存入 new_caches 列表供下一步推理用。
        return x, kv_cache

# ==========================================
# Part 5 核心组件：单个专家 MLP
# ==========================================
# 本文件定义了 MoE（混合专家）架构中的"专家"模块。
# 每个专家就是一个独立的前馈网络（FFN），可以选用 SwiGLU 或 传统 GELU 两种激活方案。
#
# MoE 的直觉类比：
#   想象一个"专家会诊系统"——来了一个问题（token），路由器（Router/Gate）根据问题的
#   特征选择最合适的几位专家来回答，每位专家给出自己的意见，最后加权融合。
#   专家之间参数不共享，各自擅长不同类型的输入模式（语法、语义、事实知识等）。
#
# SwiGLU 与 GELU 的核心区别：
#   - 传统 FFN（GELU）：x → Linear↑ → GELU → Linear↓ → x
#     只有一个信息流，GELU 是一个固定的"门"函数（门控值与输入正相关）。
#   - SwiGLU FFN：x → inp1(x) ⊙ SiLU(inp2(x)) → Linear↓ → x
#     两个独立的信息流，"门"（inp2）是学习出来的，不再与主通路（inp1）绑定。
#     ⊙ 表示逐元素相乘（Hadamard 乘积），SiLU(x) = x * sigmoid(x)。
#     直观理解：inp2 输出一个 0~1 的软开关，控制 inp1 的哪些信息可以通过。
#     这个"开关"是可训练的，因此比 GELU 的固定开关更灵活、表达能力更强。
#     LLaMA、Mistral、GPT-4 均采用 SwiGLU。
#
# 与 Part 3 model_modern.py 中 SwiGLU FFN 的关系：
#   Part 3 的 FFN 是一个整体，所有 token 都经过同一个 FFN。
#   Part 5 的 ExpertMLP 将 FFN 拆成多个"专家"，每个 token 只激活少数几位专家，
#   从而实现"条件计算"——不同 token 走不同的参数路径，计算量不变但参数量大幅提升。
from __future__ import annotations
import torch.nn as nn


# ==========================================
# ExpertMLP：单个专家 MLP（支持 SwiGLU / GELU 双模式）
# ==========================================
class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU)."""
    def __init__(self, dim: int, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        # ==========================================
        # 构造函数参数说明
        # ==========================================
        #   dim    : 模型隐藏维度（输入和输出的维度相同），如 512。
        #            ExpertMLP 是残差连接内部的分支，所以输入 dim 维，输出也是 dim 维。
        #   mult   : 扩展倍数。中间层的维度 = dim * mult = 4 * dim（如 512→2048）。
        #            为什么需要扩展？FFN 的核心思想是"先升维增加容量，再降维回原尺寸"。
        #            扩展倍数 4 是 Transformer 论文的标准值，在计算开销与模型容量之间取平衡。
        #   swiglu : True=使用 SwiGLU 激活，False=使用传统 GELU FFN。
        #            SwiGLU 的参数量比 GELU 多约 33%（多了一个 Linear(dim, inner)），
        #            但表达能力更强。为了参数公平对比，SwiGLU 通常把 mult 从 4 降到 8/3≈2.67
        #            （LLaMA 的做法），本实现简化处理，两种模式都用同一 mult。
        #   dropout: Dropout 概率。0.0 = 不使用 dropout。
        #            现代大模型训练趋向于设 0 或极小的值（数据量大，不过拟合）。
        super().__init__()
        # inner = 扩展后的中间层维度
        inner = mult * dim

        if swiglu:
            # ══════════════════════════════════════════════
            # SwiGLU 分支：激活门控 FFN
            # ══════════════════════════════════════════════
            # SwiGLU 的前向传播公式：
            #   y = out( inp1(x) ⊙ SiLU(inp2(x)) )
            # 分解理解：
            #   1. inp1(x)：主信息通路，投影到 inner 维（dim → inner）
            #   2. inp2(x)：门控通路，同样投影到 inner 维（dim → inner），
            #      然后经过 SiLU 激活函数（sigmoid 加权的线性单元）
            #   3. inp1 ⊙ SiLU(inp2)：门控信号按元素"过滤"主信息
            #      → 门控值接近 1 的位置，主信息畅通无阻
            #      → 门控值接近 0 的位置，主信息被抑制
            #   4. out(...)：投影回 dim 维（inner → dim），恢复原始形状
            #
            # bias=False：LLaMA 的设计选择。去掉偏置项一方面减少参数量，
            # 另一方面偏置在 RMSNorm（均值为 0）后作用有限。

            # 主信息通路：dim → inner（如 512 → 2048）
            self.inp1 = nn.Linear(dim, inner, bias=False)
            # 门控通路：dim → inner，独立可训练的投影矩阵
            self.inp2 = nn.Linear(dim, inner, bias=False)
            # SiLU 激活：SiLU(x) = x * sigmoid(x)
            # 与 Swish 相同，比 ReLU 更平滑（没有"死神经元"问题），
            # 比 GELU 计算更快（不需要误差函数的近似）
            self.act = nn.SiLU()
            # 输出投影：inner → dim（如 2048 → 512），回到残差路径
            self.out = nn.Linear(inner, dim, bias=False)
            # Dropout 层：训练时随机丢弃一部分输出，防止过拟合
            self.drop = nn.Dropout(dropout)
            self.swiglu = True
        else:
            # ══════════════════════════════════════════════
            # GELU 分支：传统激活 FFN（Part 2 的架构）
            # ══════════════════════════════════════════════
            # 公式：y = Dropout( Linear↓( GELU( Linear↑(x) ) ) )
            # 只有一条信息通路，GELU 是固定的非线性变换（不是可学习的门）。
            #
            # 语法：nn.Sequential 是一个"模块流水线"容器。
            # forward 时，数据依次流过列表中的每个子模块，就像一个组装好的流水线。
            # 与普通 Python list 的区别：
            #   nn.Sequential 是 nn.Module 的子类，所以可以 .to(device)、.parameters()、.train()。
            #   普通 list 没有这些方法，放入 list 的子模块不会被 PyTorch 自动追踪。
            self.ff = nn.Sequential(
                nn.Linear(dim, inner),          # 升维：dim → inner
                nn.GELU(),                       # 固定非线性激活
                nn.Linear(inner, dim),           # 降维：inner → dim
                nn.Dropout(dropout)              # 随机丢弃
            )
            self.swiglu = False

    def forward(self, x):
        # ==========================================
        # forward：专家前向传播
        # ==========================================
        # 输入 x 的形状：(B, T, dim)
        #   B = batch size（批次大小）
        #   T = sequence length（token 序列长度）
        #   dim = 隐藏维度
        # 注意：MoE 路由时，输入可能只是被分配给该专家的那些 token 的子集，
        # 此时 T 是"被分配给本专家的 token 数"，而不是全序列长度。
        # 输出形状：(B, T_subset, dim)，与输入形状相同。
        if self.swiglu:
            # ─── SwiGLU 前向：两条通路并行计算，最后逐元素融合 ───
            # a 形状：(B, T, inner)，主信息的原始投影
            a = self.inp1(x)
            # b 形状：(B, T, inner)：
            #   inp2(x) → 线性投影到 inner 维
            #   SiLU(...) → 非线性门控值，范围约 [-0.28, +∞)
            #   这里的";"分号语法只是在一行写多条语句（Python 允许），
            #   a 和 b 的计算互不依赖，GPU 可以并行执行这两个 matmul。
            b = self.act(self.inp2(x))

            # a * b：逐元素相乘（Hadamard 乘积），形状 (B, T, inner)
            #   → 门控值 b 逐位置"筛选"主信息 a
            # out(...)：投影回 dim 维，(B, T, inner) → (B, T, dim)
            # drop(...)：训练时随机置零部分输出，推理时自动关闭
            return self.drop(self.out(a * b))
        else:
            # ─── GELU 前向：单通路顺序执行 ───
            # nn.Sequential 自动按顺序调用各层，代码简洁但灵活性较低。
            return self.ff(x)
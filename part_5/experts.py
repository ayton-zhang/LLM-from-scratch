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

# 语法：`from __future__ import annotations`
#   Python 3.7+ 引入的特性，将所有类型注解（如 `dim: int`）推迟求值（PEP 563）。
#   好处：(1) 允许前向引用——类的方法可以用自己的类名做类型注解而不会报 NameError；
#         (2) 导入速度更快——注解在定义时不求值，只在需要时才解析。
#   这里实际上不需要（代码中没有自引用类型），但作为现代 Python 习惯保留。
from __future__ import annotations
import torch.nn as nn


# ==========================================
# ExpertMLP：单个专家 MLP（支持 SwiGLU / GELU 双模式）
# ==========================================
class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU).

    为什么 MoE 中的"专家"就是 MLP 而非更复杂的结构？
      1. Transformer 中的知识主要存储在 FFN 子层中（研究表明注意力层负责"检索"，
         FFN 层负责"存储"事实和模式），用 FFN 做专家能最大化知识容量。
      2. MLP 的计算量可控（matmul 为主，GPU 友好），适合大规模并行。
      3. 简洁的模块边界使得路由和负载均衡更容易实现。
    """

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
        #   dropout: Dropout 概率，取值范围 [0, 1)。0.0 = 不使用 dropout。
        #            现代大模型训练趋向于设 0 或极小的值（如 0.1）：
        #               - 数据量巨大时自然防止过拟合，不需要强正则化
        #               - dropout 会破坏残差流的信息连续性，影响深层模型训练稳定性
        #            类比：dropout 像考试时随机蒙住部分知识点——逼你全面掌握，不能死记硬背。
        #                  但不考试（推理）时就不蒙了（自动关闭）。

        # 语法：super().__init__()
        #   调用父类 nn.Module 的构造函数，注册子模块追踪机制。
        #   只有经过 super().__init__() 之后，赋值给 self 的 nn.Module 子对象才会被
        #   PyTorch 自动追踪（出现在 .parameters()、.to(device)、.state_dict() 中）。
        super().__init__()
        # inner = 扩展后的中间层维度
        inner = mult * dim

        if swiglu:
            # ══════════════════════════════════════════════
            # SwiGLU 分支：激活门控 FFN
            # ══════════════════════════════════════════════
            # SwiGLU 的前向传播公式：
            #   y = out( inp1(x) ⊙ SiLU(inp2(x)) )
            # 分解理解（四步流水线）：
            #   1. inp1(x)：主信息通路，投影到 inner 维（dim → inner）
            #   2. inp2(x)：门控通路，同样投影到 inner 维（dim → inner），
            #      然后经过 SiLU 激活函数（sigmoid 加权的线性单元）
            #   3. inp1 ⊙ SiLU(inp2)：门控信号按元素"过滤"主信息
            #      → 门控值接近 1 的位置，主信息畅通无阻
            #      → 门控值接近 0 的位置，主信息被抑制
            #   4. out(...)：投影回 dim 维（inner → dim），恢复原始形状
            #
            # 为什么用两条独立的通路而非 GELU 的一条？
            #   GELU 的"门"由输入自身决定（x * Φ(x)），信息内容和门控信号耦合在一起。
            #   SwiGLU 的"门"（inp2）有自己独立的可训练权重，可以学会"基于什么条件来决定
            #   哪些信息该通过"，而不受限于主信息通路的内容。这种解耦带来了更强的表达能力。
            #
            # bias=False：LLaMA 的设计选择。去掉偏置项一方面减少参数量，
            # 另一方面偏置在 RMSNorm（均值为 0）后作用有限。

            # 主信息通路：dim → inner（如 512 → 2048）
            self.inp1 = nn.Linear(dim, inner, bias=False)
            # 门控通路：dim → inner，独立可训练的投影矩阵
            #   注意：inp1 和 inp2 是两个独立的 Linear 层，参数不共享。
            #   这使得"判定条件"（门控信号来源）和"被判定内容"（主信息）可以有不同的投影方式。
            self.inp2 = nn.Linear(dim, inner, bias=False)
            # SiLU 激活：SiLU(x) = x * sigmoid(x)
            # 与 Swish 相同，比 ReLU 更平滑（没有"死神经元"问题），
            # 比 GELU 计算更快（不需要误差函数的近似）。
            # 直观理解：SiLU 像一个"软开关"——
            #   x > 0  时，sigmoid(x) ≈ 1，SiLU(x) ≈ x（线性通过）
            #   x < 0  时，sigmoid(x) ≈ 0，SiLU(x) ≈ 0（被抑制）
            #   x ≈ 0  时，sigmoid(x) ≈ 0.5，SiLU(x) ≈ 0.5x（部分通过）
            #   最妙的是 x 可以略小于 0（最小值约 -0.28），形成微弱的"负门控"，比 ReLU 丰富。
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
            # GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数。
            #   直觉：GELU 以概率 Φ(x) 让 x 通过、概率 1-Φ(x) 丢弃——像一个"软 dropout"。
            #
            # 语法：nn.Sequential 是一个"模块流水线"容器。
            # forward 时，数据依次流过列表中的每个子模块，就像一个组装好的流水线。
            # 与普通 Python list 的关键区别：
            #   nn.Sequential 是 nn.Module 的子类 → 可以 .to(device)、.parameters()、.train()。
            #   普通 list 没有这些方法，放入 list 的子模块不会被 PyTorch 自动追踪。
            self.ff = nn.Sequential(
                nn.Linear(dim, inner),          # 升维：dim → inner（4×容量扩展）
                nn.GELU(),                       # 固定非线性激活（软 dropout 式的门控）
                nn.Linear(inner, dim),           # 降维：inner → dim（恢复残差路径尺寸）
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
        # 这是 MoE 的核心效率来源——每个专家只处理被路由到的 token，
        # 计算量 = O(激活token数 × 专家参数) 而非 O(全部token × 全部专家参数)。
        # 输出形状：(B, T_subset, dim)，与输入形状相同（残差连接要求）。
        if self.swiglu:
            # ─── SwiGLU 前向：两条通路并行计算，最后逐元素融合 ───

            # 步骤 1a：主信息通路
            # 形状变换：(B, T, dim) → (B, T, inner)
            # 例如：(2, 16, 512) → (2, 16, 2048)（当 mult=4 时）
            a = self.inp1(x)

            # 步骤 1b：门控通路（与步骤 1a 并行，无依赖关系）
            # 形状变换：(B, T, dim) → (B, T, inner)
            # 先线性投影到 inner 维，再经 SiLU 激活产生"软开关"信号
            # inp2 和 inp1 是两个独立的线性层——GPU 可以同时执行这两个矩阵乘法
            b = self.act(self.inp2(x))

            # 步骤 2：门控融合
            # a * b：逐元素相乘（Hadamard 乘积，不是矩阵乘法！）
            #   形状 (B, T, inner) ⊙ (B, T, inner) → (B, T, inner)
            #   语义：门控值 b 逐位置"筛选"主信息 a——b 趋向 0 的位置，a 的对应元素被抑制
            # 步骤 3：输出投影 + dropout
            #   out(...)：形状 (B, T, inner) → (B, T, dim)，回到残差路径尺寸
            #   drop(...)：训练时随机置零部分输出，推理时 nn.Dropout 自动变为恒等映射
            return self.drop(self.out(a * b))
        else:
            # ─── GELU 前向：单通路顺序执行 ───
            # nn.Sequential 自动按顺序调用各层，等价于手写：
            #   x = Linear(dim,inner)(x)
            #   x = GELU()(x)
            #   x = Linear(inner,dim)(x)
            #   x = Dropout(dropout)(x)
            # 代码简洁但灵活性较低——无法在中间插入旁路或分支操作。
            return self.ff(x)

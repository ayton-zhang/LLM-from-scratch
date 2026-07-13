# ==========================================
# Part 5 混合前馈网络（HybridFFN）：稠密 + 专家 的融合模块
# ==========================================
#
# 这是 Part 5 最核心的创新设计——将传统的稠密 FFN 与 MoE 并行组合，
# 用一个可调节的 α 系数在"稳定性"和"容量"之间做权衡。
#
# 为什么需要 HybridFFN？
#   MoE 的优势是参数量大、容量高，但也有两大风险：
#     1. 专家坍塌（Expert Collapse）：门控可能把所有 token 都发给同一两个专家。
#     2. 训练不稳定：稀疏路由的梯度信号弱，早期训练时门控可能"乱指"。
#   保留一条稠密 FFN 通路作为"安全网"——
#     即使 MoE 路由失败，稠密 FFN 仍能保证每个 token 得到基本处理，
#     相当于给模型装了一个"永不掉线"的备用引擎。
#
# 与 MoE（moe.py）的关系：
#   MoE 是纯专家模块，所有 token 经过门控→专家→加权融合。
#   HybridFFN 在 MoE 旁边并联了一条稠密 FFN，二者的输出按 α 加权混合。
#   类比：MoE 像"专家会诊"（少数精英各抒己见），
#         稠密 FFN 像"全科门诊"（人人经过统一处理），
#         HybridFFN = α × 全科门诊 + (1-α) × 专家会诊。
#
# 本模块的作用范围：
#   HybridFFN 替换了标准 Transformer Block 中的 FFN 子层，
#   与注意力子层通过残差连接串联。结构如下：
#     x → RMSNorm → Attention → 残差相加
#        → RMSNorm → HybridFFN  → 残差相加 → 输出
#   即 HybridFFN 的位置与传统 FFN 完全相同，只是内部结构从单路变成双路融合。

# 语法：`from __future__ import annotations`
#   Python 3.7+ 的特性（PEP 563），将所有类型注解推迟求值。
#   好处：(1) 允许前向引用——类的方法可以用自己的类名做类型注解而不会报 NameError；
#         (2) 导入速度更快——注解在定义时不求值，只在需要时才解析。
from __future__ import annotations
import torch.nn as nn
from moe import MoE


# ==========================================
# HybridFFN：α-混合前馈网络（稠密 FFN + MoE 并联）
# ==========================================
class HybridFFN(nn.Module):
    """Blend dense FFN with MoE output: y = α * Dense(x) + (1−α) * MoE(x).
    Use α∈[0,1] to trade between stability (dense) and capacity (MoE).

    核心公式（极其简单但极其有效）：
        y = α × DenseFFN(x) + (1 - α) × MoE(x)

    两条通路的角色分工：
      - DenseFFN（稠密通路）：传统的升维→激活→降维 MLP，所有 token 都经过。
        职责是提供"稳定基线"——无论 MoE 路由好坏，至少有一条可靠的信号通路。
      - MoE（专家通路）：门控路由 → 专家处理 → 加权融合，参数量巨大但每个 token
        只激活 k/N_expert 的专家。职责是提供"差异化能力"——不同 token 可以走
        不同的参数路径，模型容量远大于稠密 FFN。

    α 参数的含义与调节：
      - α = 1.0：纯稠密 FFN，退化为 Part 3 的标准 Transformer。最稳定，容量最低。
      - α = 0.0：纯 MoE，完全依赖专家路由。容量最高，有训练不稳定风险。
      - α = 0.5：两路均衡。推荐起步值，同时享受两种通路的优点。
      - 典型策略：训练初期设 α 较大（如 0.7~0.9），让模型先建立稳定的基础表示；
        随着训练推进逐步降低 α（如降至 0.3~0.5），释放 MoE 的容量优势。
        这种做法类似于"课程学习"——先打好基础，再扩展能力。

    辅助损失 (aux_loss)：
      MoE 模块在 forward 时会返回 aux_loss（负载均衡损失），HybridFFN 将其
      透传出去，由上层的训练循环累加到总损失中，用于防止专家坍塌。
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 0.5,
        mult: int = 4,
        swiglu: bool = True,
        n_expert: int = 4,
        k: int = 1,
        dropout: float = 0.0,
    ):
        # ==========================================
        # 构造函数参数说明
        # ==========================================
        #   dim       : 隐藏维度（输入/输出维度相同，适配残差连接），如 512。
        #   alpha     : 稠密通路权重 α ∈ [0, 1]。
        #               α 越大 → 越偏向稠密 FFN → 更稳定但容量受限。
        #               α 越小 → 越偏向 MoE    → 容量更大但训练更依赖辅助损失。
        #   mult      : FFN 中间层扩展倍数，默认 4（标准 Transformer 值）。
        #               中间层维度 = dim × mult（如 512 × 4 = 2048）。
        #   swiglu    : True = 稠密 FFN 和专家都用 SwiGLU 激活（LLaMA 风格）；
        #               False = 都用传统 GELU 激活（GPT-2 风格）。
        #               注意：稠密和 MoE 使用相同的 swiglu 设定，保持一致。
        #   n_expert  : 专家总数。每个 token 只会激活其中 k 个专家，
        #               所以计算量 ≈ 稠密 FFN 的 k 倍（而非 n_expert 倍）。
        #               典型值：4~8（小型 MoE），8~16（中型），16+（大型）。
        #   k         : 每个 token 激活的专家数（top-k 路由）。
        #               k=1（Switch Transformer 风格）：最稀疏，每个 token 只走 1 个专家。
        #               k=2（Mixtral 风格）：略有冗余，容错性更好。
        #   dropout   : Dropout 概率，范围 [0, 1)。0 = 不使用。
        #               MoE 模型参数量大但每个 token 实际走的参数量少（稀疏激活），
        #               相比同参数量的稠密模型，过拟合风险更低，dropout 可以设得更小。

        # 语法：super().__init__()
        #   调用父类 nn.Module 的构造函数，注册子模块追踪机制。
        #   只有经过这一步后，赋值给 self 的 nn.Module 子对象才会被 PyTorch
        #   自动追踪——出现在 .parameters()、.state_dict()、.to(device) 中。
        super().__init__()

        # ─── α 混合系数（不是 nn.Parameter，不会参与梯度更新） ───
        # α 是一个普通的 Python float，保存在 self 上作为超参数。
        # 如果想学习最优 α，可以改成：
        #   self.alpha = nn.Parameter(torch.tensor(alpha))
        # 但固定 α 更简单、更可控，训练初期也避免了 α 被推到极端值的风险。
        self.alpha = alpha

        # inner = 中间层扩展维度（稠密 FFN 和 MoE 共用此值）
        # 例如 dim=512, mult=4 → inner=2048
        inner = mult * dim

        # ═══════════════════════════════════════════════════════
        # 通路一：稠密 FFN（"全科门诊"）
        # ═══════════════════════════════════════════════════════
        # 使用 nn.Sequential 搭建经典的三步流水线：
        #   Linear↑ (dim → inner) → GELU 激活 → Linear↓ (inner → dim) → Dropout
        #
        # 注意：这里用的是 GELU 而非 swiglu 参数——当前版本固定稠密通路用 GELU，
        # 而 MoE 通路通过 swiglu 参数控制。设计动机：
        #   稠密通路作为"稳定基线"，用朴素 GELU 更简单可靠；
        #   MoE 通路作为"能力扩展"，用 SwiGLU 更强的非线性。
        # 如果希望两路一致，可以将 GELU 替换为条件选择的 SiLU+gate 结构。
        #
        # 语法：nn.Sequential 是一个"模块流水线"容器。
        # forward 时数据依次流过其中每个子模块，像工厂流水线一样。
        # 与普通 Python list 的关键区别：nn.Sequential 是 nn.Module 子类 →
        #   .to(device)、.parameters()、.train() 等调用会自动递归到所有子模块。
        #   普通 list 没有这个能力——丢进去的子模块不会被 PyTorch 追踪。
        self.dense = nn.Sequential(
            nn.Linear(dim, inner),   # 升维：dim → inner（如 512→2048），增加容量
            nn.GELU(),               # GELU 非线性激活（"软门控"，见 experts.py 注释）
            nn.Linear(inner, dim),   # 降维：inner → dim（2048→512），恢复残差路径尺寸
            nn.Dropout(dropout),     # 随机丢弃部分输出，防止过拟合
        )

        # ═══════════════════════════════════════════════════════
        # 通路二：MoE（"专家会诊"）
        # ═══════════════════════════════════════════════════════
        # MoE 类（moe.py）内部包含：
        #   1. TopKGate（门控）：为每个 token 选出 top-k 专家及权重
        #   2. n_expert 个 ExpertMLP（专家）：每个专家是独立的 SwiGLU/GELU FFN
        # 前向时：gate 做路由决策 → dispatch（分发 token 到对应专家）→
        #         专家并行计算 → combine（用门控权重加权融合专家输出）
        #
        # MoE 的前向返回 (输出, aux_loss) 二元组：
        #   - 输出形状：(B, T, dim)，与输入形状相同
        #   - aux_loss：标量，负载均衡辅助损失
        self.moe = MoE(
            dim,
            n_expert=n_expert,
            k=k,
            mult=mult,
            swiglu=swiglu,
            dropout=dropout,
        )

    def forward(self, x):
        # ==========================================
        # forward：双路计算 → α 加权融合
        # ==========================================
        # 输入 x 的形状：(B, T, dim)
        #   B = batch size（批次大小，一次处理几个样本）
        #   T = sequence length（token 序列长度）
        #   dim = 隐藏维度
        # 输出 y 的形状：(B, T, dim)，与输入完全相同（残差连接要求）
        #
        # 数据流动（双路并行）：
        #
        #                  ┌→ dense(x) ──────────────────→ y_dense ─┐
        #   x (B, T, dim) ─┤                                        ├→ α·y_dense + (1-α)·y_moe → y
        #                  └→ moe(x) → (y_moe, aux_loss) ──────────┘
        #
        # 为什么两路可以"并联"而非"串联"？
        #   串联（先 dense 再 moe 或反过来）会把两路的信息处理顺序绑定，
        #   训练时误差信号需要穿过两条路径才能到达输入，梯度衰减更严重。
        #   并联让每条通路有独立的梯度路径——
        #     dense 通路直接学习"所有 token 通用的模式"，
        #     moe  通路直接学习"不同 token 的差异化特征"，
        #   分工明确、梯度通畅。

        # ─── 步骤 1：稠密 FFN 前向 ───
        # nn.Sequential 自动依次调用各层，等价于：
        #   y_dense = Dropout(Linear↓(GELU(Linear↑(x))))
        # 形状变换：(B, T, dim) → (B, T, inner) → (B, T, inner) → (B, T, dim) → (B, T, dim)
        # 注意：dense 通路不产生 aux_loss，所有 token 无差别通过
        y_dense = self.dense(x)

        # ─── 步骤 2：MoE 前向 ───
        # moe.forward(x) 内部流程：
        #   展平 token → 门控选专家 → 分发→专家计算→加权回归 → 恢复形状
        # 语法：`y_moe, aux = self.moe(x)` 是元组解包。
        #   moe.forward() 返回 (专家输出, 辅助损失) 二元组，
        #   Python 自动把两个返回值分别绑定到 y_moe 和 aux。
        #   y_moe 形状：(B, T, dim)，与输入相同
        #   aux 形状：标量（0 维张量），负载均衡损失值
        y_moe, aux = self.moe(x)

        # ─── 步骤 3：α 加权融合 ───
        # 公式：y = α × y_dense + (1 - α) × y_moe
        #
        # 这是逐元素加法（不是矩阵乘法！），广播规则：
        #   α 和 (1-α) 是标量，自动广播到 (B, T, dim) 的每一个元素。
        #   等价于对每个 token 的每一个隐藏维度做线性插值：
        #     y[b, t, c] = α × y_dense[b, t, c] + (1-α) × y_moe[b, t, c]
        #
        # 直观理解（类比音响调音台）：
        #   α 是"推子"——推到 1，只听稠密 FFN（清汤寡水但稳定）；
        #   拉到 0，只听 MoE（层次丰富但有杂音风险）；中间位置则两种风味都有一点。
        #
        # 为什么 α + (1-α) = 1（凸组合）？
        #   凸组合保证输出 y 的数值范围不会超出 y_dense 和 y_moe 的数值范围，
        #   避免信号幅度在混合过程中被放大或衰减。
        #   如果 α 可能 >1 或 <0，输出幅度会失控，残差连接后可能爆炸。
        y = self.alpha * y_dense + (1.0 - self.alpha) * y_moe

        # ─── 返回：混合输出 + 辅助损失 ───
        # 上游调用方（如 Block 或训练循环）需要：
        #   y：加到残差路径上，继续向后传播
        #   aux：累加到总损失中，用于反向传播优化门控参数
        # 注意：即使 α=1.0（纯稠密），moe 前向仍会执行（产生 aux_loss），
        # 只是其输出被 (1-α)=0 乘以 0 后对 y 没有贡献了。
        # 如果确定不需要 MoE，建议直接使用标准 FFN（如 Part 3）以避免无效计算。
        return y, aux

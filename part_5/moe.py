# ==========================================
# Part 5 混合专家层（MoE）：Token 级 Top-K 路由的调度中心
# ==========================================
#
# 本文件是 Part 5 的核心调度器（Orchestrator），负责连接"门控路由"和"专家网络"两个子系统。
# 它的职责是：
#   1. 将输入的 token 序列展平，交给门控模块（TopKGate）做路由决策
#   2. 根据路由结果，把每个 token 分发（dispatch）到对应的专家
#   3. 收集所有专家的输出，按门控权重加权融合（combine）
#   4. 恢复原始张量形状，返回融合后的输出和辅助损失
#
# MoE 架构的核心论文：
#   - Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
#   - Fedus et al. (2021): "Switch Transformers" (top-1 路由)
#   - Jiang et al. (2024): "Mixtral of Experts" (top-2 路由，SwiGLU 专家)
#
# 本实现的定位：
#   这是一个"教学友好型"实现——显式地在 Python 层面用 for 循环遍历专家，
#   而非使用 scatter/gather 等底层 CUDA 操作。这样做的好处是代码清晰、易于理解，
#   代价是单 GPU 上不如高度优化的 MoE 内核快（但足够用于学习和实验）。
#   生产级实现（如 MegaBlocks、Tutel）会用块稀疏矩阵乘法来并行化专家计算。
#
# 与其他模块的关系：
#   - gating.py    → TopKGate  门控决策（选哪些专家、权重多少）
#   - experts.py   → ExpertMLP 单个专家（独立的 SwiGLU/GELU FFN）
#   - moe.py       → MoE       调度器（本文件，把门控和专家串起来）
#   - block_hybrid.py → HybridFFN 在 MoE 旁并联稠密 FFN（更上层的集成模块）

# 语法：`from __future__ import annotations`
#   Python 3.7+ 的特性（PEP 563），将所有类型注解推迟求值。
#   好处：(1) 允许前向引用——类的方法可以用自己的类名做类型注解而不会报 NameError；
#         (2) 导入速度更快——注解在定义时不求值，只在需要时才解析。
from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

# ==========================================
# MoE：混合专家层调度器
# ==========================================
class MoE(nn.Module):
    """Mixture‑of‑Experts layer (token‑wise top‑k routing).
    Implementation is single‑GPU friendly (loops over experts for clarity).
    https://arxiv.org/pdf/2101.03961

    核心流程（六步走）：
      第一步：展平 token → 把 (B, T, C) 压成 (B*T, C)，让门控像看待独立样本一样看待每个 token
      第二步：门控路由 → TopKGate 为每个 token 选出 top-k 个专家及对应权重
      第三步：分发 (dispatch) → 把 token 按专家编号分组，准备专家计算
      第四步：专家计算 → 每个专家独立处理分配给它的 token 子集
      第五步：合并 (combine) → 用门控权重对各专家输出做加权求和
      第六步：恢复形状 → (B*T, C) 变回 (B, T, C)，交还给残差路径

    与 HybridFFN（block_hybrid.py）的关系：
      MoE 是"纯专家模块"，所有输出都来自路由后的专家计算。
      HybridFFN 在 MoE 旁边并联了一条稠密 FFN，用 α 系数混合两者的输出，
      在"稳定性"和"容量"之间做权衡。详见 block_hybrid.py 的注释。
    """

    def __init__(self, dim: int, n_expert: int, k: int = 1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        # ==========================================
        # 构造函数参数说明
        # ==========================================
        #   dim       : 隐藏维度（输入/输出维度相同，适配残差连接），如 512。
        #   n_expert  : 专家总数。每个 token 只激活其中 k 个，因此计算量 ≈ 稠密 FFN 的 k 倍，
        #              而非 n_expert 倍——这是 MoE"参数多但计算少"的核心秘密。
        #              Mixtral 8×7B 用了 n_expert=8，但 k=2，实际每次只走 2/8 的参数。
        #   k         : 每个 token 激活的专家数。k=1（Switch Transformer）：最稀疏、最快；
        #              k=2（Mixtral）：略有冗余，一个专家"没空"时另一个顶上。k 通常远小于 n_expert。
        #   mult      : FFN 中间层扩展倍数，默认 4。和标准 Transformer FFN 的规则一样。
        #   swiglu    : True = 每个专家内部用 SwiGLU 激活（LLaMA/Mixtral 风格）；
        #              False = 用传统 GELU 激活（GPT-2 风格）。
        #   dropout   : Dropout 概率，范围 [0, 1)。MoE 因稀疏激活自带正则化效果，
        #              dropout 可以设得比稠密模型更小（默认 0.0）。

        # 语法：super().__init__()
        #   调用父类 nn.Module 的构造函数，注册子模块追踪机制。
        #   只有经过这一步后，赋值给 self 的 nn.Module 子对象才会被 PyTorch
        #   自动追踪——出现在 .parameters()、.state_dict()、.to(device) 中。
        super().__init__()

        # 保存配置为实例属性，方便后续 forward 中引用
        self.dim = dim
        self.n_expert = n_expert
        self.k = k

        # ═══════════════════════════════════════════════════════
        # 组件一：门控路由器（TopKGate）
        # ═══════════════════════════════════════════════════════
        # TopKGate 是一个轻量级模块（只有一个 Linear 层），负责为每个 token
        # 计算对 n_expert 个专家的"偏好分数"，并通过 softmax + topk 选出
        # 最合适的 k 个专家。forward 返回 (专家索引, 门控权重, 辅助损失) 三元组。
        #
        # 门控的参数是模型要学习的——它会逐渐学会哪些 token 该去找哪些专家。
        # 类比：门控像一个"分诊台护士"，根据每个病人的症状（token 隐状态）决定挂哪个科室。
        self.gate = TopKGate(dim, n_expert, k=k)

        # ═══════════════════════════════════════════════════════
        # 组件二：专家集合（n_expert 个 ExpertMLP）
        # ═══════════════════════════════════════════════════════
        # 每个专家是一个独立的 FFN，有自己的一套参数（权重矩阵）。
        # 专家之间参数完全不共享——这是 MoE 参数量大的根本原因。
        #
        # 语法：nn.ModuleList([...]) 创建的是 PyTorch 模块容器。
        #   与普通 Python list 的关键区别：
        #     nn.ModuleList 是 nn.Module 的子类，里面的子模块会被 PyTorch 自动追踪，
        #     因此 .parameters()、.to(device)、.state_dict() 等调用会递归到每个专家。
        #     普通 list 没有这个能力——丢进去的模块不会被参数优化器（optimizer）看到。
        #   注：nn.ModuleList 不实现 forward，它只是容器，forward 需要自己写循环。
        #
        # 列表推导式：[ExpertMLP(...) for _ in range(n_expert)] 创建 n_expert 个独立专家。
        # _ 是 Python 惯例，表示"循环变量我不需要"——只需重复 n_expert 次，不关心索引值。
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)]
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, C) → y: (B, T, C), aux_loss
        Steps: flatten tokens → gate → per‑expert forward → scatter back with weights.

        参数:
            x : 形状 (B, T, C)
                B = batch_size（批次大小）
                T = sequence_length（token 序列长度）
                C = dim（隐藏维度）
        返回:
            y    : 形状 (B, T, C)，MoE 层的输出，与输入形状相同（残差连接要求）
            aux  : 标量张量，负载均衡辅助损失，需要累加到训练总损失中
        """

        # ──────────────────────────────────────────────────
        # 步骤 1：把 token 序列展平，为 token 级路由做准备
        # ──────────────────────────────────────────────────
        # 为什么要展平？
        #   门控模块把每个 token 当作独立个体做路由决策——它不关心 token 属于哪个 batch
        #   或位于序列的哪个位置。展平后所有 token 平等地站在同一条起跑线上。
        #   类比：班级拍集体照时，先把方阵 (B, T) 拉成一条直线 (S,)，
        #         然后逐个点名分发到不同任务组（专家）——排成直线更方便点名。

        # 语法：B, T, C = x.shape 是元组解包（Tuple Unpacking）。
        #   x.shape 返回一个 torch.Size 元组（如 (2, 16, 512)），
        #   Python 把三个元素同时赋给左边的 B、T、C 三个变量。
        B, T, C = x.shape

        # S = 总 token 数 = batch_size × sequence_length
        # 例如 B=2（2 个句子）, T=16（每句 16 个 token） → S=32
        S = B * T

        # 语法：x.reshape(S, C) 改变张量形状，不改变数据。
        #   形状变换：(B, T, C) → (S, C)，即 (2, 16, 512) → (32, 512)
        #   展平后每一行就是一个独立的 token，等待门控"点名分配"。
        #   与 .view() 的区别：.reshape() 在内存不连续时会先拷贝再变形（更安全），
        #   .view() 要求张量内存连续，否则报错。这里用 reshape 更稳妥。
        x_flat = x.reshape(S, C)              # (B,T,C) → (S,C)

        # ═══════════════════════════════════════════════════════
        # 步骤 2：门控路由——为每个 token 选出 top-k 专家
        # ═══════════════════════════════════════════════════════
        # 调用 TopKGate.forward(x_flat)，内部做了三件事：
        #   a) Linear 投影 x_flat → logits (S, E)，计算每个 token 对 E 个专家的偏好得分
        #   b) Softmax 归一化 → 概率分布
        #   c) TopK 选出前 k 个专家，返回索引和权重
        # 同时计算负载均衡辅助损失（aux），防止所有 token 都涌向同一个专家。
        #
        # 语法：`idx, w, aux = self.gate(x_flat)` 是元组解包（三元组解包）。
        #   gate.forward() 返回 (indices, weights, aux_loss) 三个值，
        #   Python 一次性把它们绑定到 idx、w、aux 三个变量。
        #   返回形状：
        #     idx → (S, k) long   类型，每个 token 的 k 个专家编号
        #     w   → (S, k) float 类型，对应专家的门控权重
        #     aux → 标量（0 维张量），负载均衡损失
        idx, w, aux = self.gate(x_flat)       # idx: (S,k), w: (S,k)

        # ═══════════════════════════════════════════════════════
        # 步骤 3+4：分发 (dispatch) + 专家计算（循环实现）
        # ═══════════════════════════════════════════════════════
        # 这里用一个双重循环实现分发 + 计算：
        #   外层循环：遍历每个专家 e（0 ~ n_expert-1）
        #   内层循环：遍历每个路由槽位 slot（0 ~ k-1），检查"专家 e 是否被某个 token
        #            在槽位 slot 上选中了"
        #
        # 为什么用循环而非更炫的操作？
        #   这是"教学友好型"实现——显式循环让 dispatch/scatter 逻辑一目了然。
        #   生产级实现会用 scatter/gather 或块稀疏矩阵乘法，在 GPU 上批量处理，
        #   但对理解 MoE 的核心流程来说，循环是最清晰的。
        #   实际上，单个 GPU 上专家数不多（4~16）时，这种循环的性能损失并不大。
        #
        # 关于"槽位"(slot)的概念：
        #   每个 token 被分配了 k 个专家（k 个 slot）。例如 k=2 时：
        #     槽位 0（slot=0）：首选专家（门控权重最高的那个）
        #     槽位 1（slot=1）：次选专家（门控权重第二高的那个）
        #   每个槽位上，专家 e 可能被零个、一个或多于一个 token 选中。
        #   MoE 会把被选中的 token 收集起来，一次性喂给专家 e 做批量前向。

        # y 初始化为全零张量，形状与展平后的 x 相同：(S, C)
        # 后续各个专家的输出会"填"到 y 的对应行中（加权累加）。
        # 语法：torch.zeros_like(x_flat) 创建一个与 x_flat 形状相同、值全为 0 的张量，
        #   并且自动继承 x_flat 的 device（CPU/GPU）和 dtype（float32/float16 等），
        #   比手动指定 device 和 dtype 更安全。
        y = torch.zeros_like(x_flat)          # (S, C)，初始全零，等待各专家输出累加进来

        # ─── 外层循环：遍历每个专家 ───
        # 每个专家独立处理分配给它的 token 子集，专家之间无数据依赖，理想情况下可以并行。
        # 当前用顺序循环实现，但概念上各专家的计算是独立的。
        for e in range(self.n_expert):

            # ─── 内层循环：遍历每个路由槽位 ───
            # 同一个 token 可能在多个槽位上都选中了专家 e（虽然 top-k 保证 k 个槽位
            # 的专家编号互不相同，但不同 token 可以都选同一个专家）。
            for slot in range(self.k):

                # ── 步骤 3a：检查哪些 token 在当前槽位上选中了专家 e ──
                # idx[:, slot]：取 idx 的第 slot 列，形状 (S,)，即所有 token 在
                #   第 slot 个槽位上选中的专家编号。
                # == e：逐元素比较，返回布尔张量 sel，形状 (S,)。
                #   sel[i] = True 表示第 i 个 token 在槽位 slot 上选中了专家 e。
                #
                # 语法：(tensor == value) 是逐元素比较，返回同形的布尔张量。
                #   例如 idx[:,0] = [3,1,0,3,1]，e=3 → sel = [True,False,False,True,False]
                sel = (idx[:, slot] == e)     # (S,)，布尔掩码

                # ── 步骤 3b：如果至少有一个 token 选中了专家 e，则执行专家前向 ──
                # 语法：sel.any() 检查布尔张量中是否有任意一个 True。
                #   有 → 进入分支，做专家计算
                #   无 → 跳过，节省计算（这正是 MoE 稀疏激活的体现！）
                # 极端情况：如果某个专家自始至终没有被任何 token 选中，
                #   本次 forward 它完全不做计算——这就是"条件计算"的核心效率来源。
                if sel.any():

                    # ── 步骤 3c：用布尔掩码索引出被选中的 token ──
                    # 语法：x_flat[sel] 是布尔索引（Boolean Indexing / Masked Select）。
                    #   它从 x_flat 中选出 sel[i] == True 的那些行，返回一个子张量。
                    #   形状变化：(S, C) → (M, C)，其中 M=sel.sum()=被选中的 token 数。
                    #   例如 S=32, sel 有 8 个 True → x_e 形状为 (8, 512)。
                    #
                    #   注意：布尔索引会触发内存拷贝（从稀疏位置 gather 到连续内存），
                    #   这是 MoE 的一个固有开销——dispatch 和 combine 阶段的 scatter/gather
                    #   操作在某些情况下可能比计算本身还慢。生产级实现会尽力优化这一步。
                    x_e = x_flat[sel]         # (M, C)，被分配给专家 e 的 token 子集

                    # ── 步骤 4：专家 e 处理分配给它的 token ──
                    # experts[e](x_e) 调用 ExpertMLP.forward(x_e)：
                    #   SwiGLU 模式：inp1(x_e) ⊙ SiLU(inp2(x_e)) → Linear↓ → Dropout
                    #   GELU  模式：Linear↑ → GELU → Linear↓ → Dropout
                    # y_e 形状：(M, C)，与 x_e 相同（残差结构要求）
                    y_e = self.experts[e](x_e)  # (M, C)

                    # ═══════════════════════════════════════════════════════
                    # 步骤 5：加权散射——把专家输出按门控权重累加到 y 中
                    # ═══════════════════════════════════════════════════════
                    # 这里是 MoE 的"合并"（combine）阶段，也是整个 MoE 最精妙的一行代码。
                    #
                    # 拆解 `y[sel] += w[sel, slot:slot+1] * y_e`：
                    #
                    #   1. w[sel, slot:slot+1]：
                    #      - sel 先做行索引（布尔掩码），从 w 中选出对应 token 的行
                    #      - 然后 slot:slot+1 做列切片，只取第 slot 列（即当前槽位的权重）
                    #      - 注意 slot:slot+1 而非 slot 的原因：用切片保留最后一维
                    #        形状 (M, 1) 而非 (M,)，这样乘法时广播语义正确（列向量 × 行向量）
                    #      - 形状：(M, 1)
                    #
                    #   2. w[sel, slot:slot+1] * y_e：
                    #      - y_e 形状 (M, C)，w 子集形状 (M, 1)
                    #      - 广播规则：(M, 1) 自动沿 C 维复制为 (M, C)，然后逐元素相乘
                    #      - 效果：y_e 的每一行（一个 token 的专家输出）被对应的门控权重缩放
                    #      - 语义：权重 w 告诉我们"专家 e 对 token i 有多重要"——
                    #        权重大的专家贡献大，权重小的贡献小
                    #
                    #   3. y[sel] += ...：为什么用 += 而不是 = ？
                    #
                    #      核心原因——MoE 的数学定义就是"k 个专家输出的加权和"：
                    #        y_token = w₁ × Expert₁(x) + w₂ × Expert₂(x) + ... + wₖ × Expertₖ(x)
                    #      公式里有一个 Σ（求和），代码里就需要一次一次累加。
                    #
                    #      具体执行过程（假设 k=2，token i 选了专家 A 和 B）：
                    #
                    #        外层 for slot in range(k):   # slot=0, 1
                    #
                    #         slot=0 这轮（专家 A）：
                    #           专家 e = topk_idx[i, 0] = A    ← 取 token i 的首选专家
                    #           sel 筛选出"slot=0 时首选专家是谁"的所有 token
                    #           其中 token i 的 sel=True（因为它的首选恰好是 A）
                    #           → y[i] += w[i,0] × Expert_A(x[i])
                    #           → y[i] 目前 = w₁ × Expert_A(x[i])     ← 第一份加数
                    #
                    #         slot=1 这轮（专家 B）：
                    #           专家 e = topk_idx[i, 1] = B    ← 取 token i 的次选专家（≠A，topk 保证）
                    #           sel 筛选出"slot=1 时次选专家是谁"的所有 token
                    #           其中 token i 的 sel=True（因为它的次选恰好是 B）
                    #           → y[i] += w[i,1] × Expert_B(x[i])
                    #           → y[i] 目前 = w₁×Expert_A(x[i]) + w₂×Expert_B(x[i])  ← 两轮累加完成！
                    #
                    #      topk 保证同一个 token 的 k 个专家互不重复（topk 沿专家维度选最大 k 个），
                    #      所以不会出现"同一个 token 的同一个专家被加两遍"的问题。
                    #      += 的意义不是"防重复"，而是"完成 Σ 求和"——就像记账时逐笔累加，
                    #      第一笔写在第一行，第二笔加在同一行下面，最终得到总和。
                    #
                    #   直观类比：y 像一个"意见收集表"——
                    #     每行对应一个 token，最开始是空的（y 初始化为 0）。
                    #     slot=0：每位"首选专家"在对应 token 的那一行写下意见（加权后的向量）。
                    #     slot=1：每位"次选专家"在同一行下面追加意见，和之前的内容累加。
                    #     最终每个 token 的那一行 = k 个专家的加权综合意见。
                    #     如果用 = 覆盖，那每次只剩最后一个专家的意见，前面的全白算了。
                    y[sel] += w[sel, slot:slot+1] * y_e

        # ═══════════════════════════════════════════════════════
        # 步骤 6：恢复形状——从展平视图变回 (B, T, C)
        # ═══════════════════════════════════════════════════════
        # 形状变换：(S, C) → (B, T, C)，即 (32, 512) → (2, 16, 512)
        # 由于 y 是 x_flat.reshape(S,C) 的"输出版本"，恢复形状后，每个 token
        # 的专家融合结果会自动回到它在原序列中的位置。
        # 后续的上游模块（残差连接、下一层 Transformer Block）完全感知不到
        # 这里经历了展平→分发→合并→恢复的过程——MoE 对外接口与普通 FFN 完全一致。
        y = y.view(B, T, C)

        # ─── 返回融合输出 + 辅助损失 ───
        # 调用方（通常是 HybridFFN 或训练循环）需要：
        #   y   → 加到残差路径上（和标准 FFN 的输出用法完全相同）
        #   aux → 累加到总训练损失中（一般为 λ·aux_loss，λ 通常取 0.01 量级）
        #         通过反向传播，aux_loss 的梯度推动门控参数 w_g 向"均衡路由"方向优化
        return y, aux

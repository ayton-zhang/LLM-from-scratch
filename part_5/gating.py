# ==========================================
# Part 5 门控模块（Gating / Router）：MoE 的"调度中心"
# ==========================================
#
# 门控（Router）是 MoE 架构中最关键的组件，负责回答一个核心问题：
#   "当前这个 token 应该交给哪几个专家处理？"
#
# 工作流程：
#   输入 token 隐状态 → 线性投影到专家维度 → softmax 归一化 → 选 top-k → 输出专家编号+权重
#
# 类比：门控就像一个"分诊台护士"——
#   每个病人（token）进来，护士根据症状（隐状态）判断该挂哪个科室（专家），
#   并按紧急程度（softmax 权重）分配优先级。
#
# 本文件实现了 TopKGate，包含两个核心功能：
#   1. Top-k 软路由：为每个 token 选出最合适的 k 个专家及其权重
#   2. 负载均衡辅助损失：防止所有 token 都涌向同一两个专家（"专家坍塌"）

from __future__ import annotations
import torch, torch.nn as nn


# ==========================================
# TopKGate：Top-k Softmax 门控 + Switch 风格负载均衡损失
# ==========================================
class TopKGate(nn.Module):
    """Top‑k softmax gating with Switch‑style load‑balancing aux loss.

    核心算法（分三步）：
      第一步：线性投影 → logits，形状 (S, E)，S=token数，E=专家数
      第二步：softmax 归一化 → 概率分布，每个 token 对 E 个专家的"偏好分数"
      第三步：torch.topk 选出前 k 个最高分的专家及其权重

    负载均衡辅助损失：
      借鉴 Switch Transformer (Fedus et al., 2021) 的设计，
      惩罚"某些专家被过度使用"的情况，鼓励 token 均匀分配到各专家。

    Args:
      dim: 输入隐藏维度（每个 token 的特征向量长度）
      n_expert: 专家总数
      k: 每个 token 路由到的专家数（典型值 1 或 2）
    Returns:
      (indices, weights, aux_loss) 三元组，其中
        indices: (S, k) long 类型，每个 token 被选中的专家索引
        weights: (S, k) float 类型，对应专家的门控权重（每 token 的 k 个权重之和 ≤ 1）
        aux_loss: 标量，负载均衡惩罚项
    """

    def __init__(self, dim: int, n_expert: int, k: int = 1):
        super().__init__()

        # 参数合法性检查：k 必须在 1 和专家总数之间
        # k=1（Switch Transformer 风格）：最稀疏，每个 token 只激活 1 个专家
        # k=2（Mixtral 风格）：略有冗余，提高容错性
        assert k >= 1 and k <= n_expert

        # 专家总数（记为 E）
        self.n_expert = n_expert
        # top-k 中的 k（每个 token 激活的专家数）
        self.k = k

        # ─── 门控权重矩阵：dim → n_expert ───
        # nn.Linear(dim, n_expert, bias=True) 创建一个可学习的线性变换：
        #   输入维度 dim（token 隐状态大小）
        #   输出维度 n_expert（专家数）
        #   有偏置项 bias（让模型在学习初期可以"偏向"某些专家）
        # 这个矩阵的参数形状为 (n_expert, dim)，偏置形状为 (n_expert,)
        # 每个专家的"得分" = w_g 的第 i 行与 token 隐状态的点积 + bias[i]
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        """前向传播：计算路由决策 + 负载均衡损失。

        参数:
            x : 形状 (S, C)，其中 S = batch_size × seq_len（所有 token 展平），
                C = dim（隐藏维度）
        返回:
            (topk_idx, topk_vals, aux_loss) 三元组
        """

        # ─── 步骤 1：计算每个 token 对每个专家的"原始偏好分数" ───
        # w_g(x) 做线性变换：x @ w_g.weight.T + w_g.bias
        # 形状变换：(S, C) @ (C, E) → (S, E)
        # logits[i, j] = token i 对专家 j 的原始得分（未归一化，可正可负）
        logits = self.w_g(x)                  # (S, E)

        # ─── 步骤 2：softmax 归一化 → 概率分布 ───
        # 语法：torch.softmax(logits, dim=-1)
        #   dim=-1 表示沿最后一维（专家维度 E）做 softmax
        # 效果：把 logits 的每一行（每个 token 的 E 个得分）映射为概率分布
        #   probs[i, :] 的和 = 1.0（对每个 token 而言，所有专家的"被选中概率"之和为 1）
        # softmax 的数学公式：probs[i,j] = exp(logits[i,j]) / Σ_m exp(logits[i,m])
        probs = torch.softmax(logits, dim=-1) # (S, E)

        # ─── 步骤 3：选出 top-k 个得分最高的专家 ───
        # 语法：torch.topk(probs, k=self.k, dim=-1)
        #   沿专家维度（dim=-1）对概率排序，取前 k 个
        #   返回具名元组 (values, indices)，可直接解包
        #   topk_vals  → (S, k)，选中的 k 个专家的 softmax 概率值（即门控权重）
        #   topk_idx   → (S, k)，选中的 k 个专家的索引（整数 0 ~ E-1）
        # 注意：这里在 softmax 之后取 topk，而非在 logits 阶段取——
        #   因为我们需要的是"归一化后的概率意义"，直接对 logits 取 topk 无法得到权重
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)  # (S,k)

        # ═══════════════════════════════════════════════════════════
        # 步骤 4：计算负载均衡辅助损失（Switch Transformer 风格）
        # ═══════════════════════════════════════════════════════════
        #
        # 为什么需要这个损失？
        #   如果不加约束，门控可能学会"偷懒"——把所有 token 都发给同一两个专家。
        #   结果：少数专家过载、多数专家闲置，模型容量被严重浪费。
        #   这就是臭名昭著的"专家坍塌"（Expert Collapse）问题。
        #
        # ═══════════════════════════════════════════════════════════
        # 辅助损失的核心直觉（一步步推导）
        # ═══════════════════════════════════════════════════════════
        #
        # 我们要度量"专家使用得均不均匀"。怎么度量？
        # 直观想法：如果 4 个专家，理想情况是每个专家处理 1/4 的 token。
        # 如果某个专家处理了 100% 的 token，那就完全崩了。
        #
        # 我们构造两个"画像"来描述专家的使用情况，每个都是长度为 E 的向量：
        #
        #   importance[j] = 专家 j 的"软受欢迎程度"
        #       = 所有 token 对专家 j 的 softmax 概率的平均值
        #       = (1/S) × Σ_i softmax(logits[i])[j]
        #       解读：所有 token "觉得"专家 j 有多重要？（取值范围 [0,1]，Σ_j importance[j] = 1）
        #
        #   load[j] = 专家 j 的"硬实际负载"
        #       = 把专家 j 当作"首选"（top-1）的 token 占比
        #       = (首选专家为 j 的 token 数) / S
        #       解读：实际上有多少 token "去了"专家 j？（取值范围 [0,1]，Σ_j load[j] = 1）
        #
        # 现在有了两个概率分布，怎么判断它们"够不够均匀"？
        #
        # ─── 关键洞察：内积 Σ_j (importance[j] × load[j]) 度量了"偏好与实际的一致程度" ───
        #
        # 用具体例子理解（E=4 个专家，S=100 个 token）：
        #
        #   【情况 A：完美均衡——每个专家都均分资源】
        #     importance = [0.25, 0.25, 0.25, 0.25]   ← 每个专家软偏好都均等
        #     load       = [0.25, 0.25, 0.25, 0.25]   ← 每个专家实际承担 25% 的 token
        #     Σ importance × load = 0.25×0.25 + 0.25×0.25 + 0.25×0.25 + 0.25×0.25
        #                         = 0.0625 × 4 = 0.25
        #                       ← 这个值很小！
        #
        #   【情况 B：专家坍塌——专家 0 垄断一切】
        #     importance = [1.0,  0.0,  0.0,  0.0 ]   ← 所有 token 都觉得"只有专家 0 靠谱"
        #     load       = [1.0,  0.0,  0.0,  0.0 ]   ← 所有 token 也确实都去了专家 0
        #     Σ importance × load = 1.0×1.0 + 0×0 + 0×0 + 0×0 = 1.0
        #                       ← 这个值很大！
        #
        #   【情况 C：软偏好略偏，但硬分配还算均匀——中等状态】
        #     importance = [0.5, 0.2, 0.2, 0.1]       ← token 们更喜欢专家 0
        #     load       = [0.3, 0.3, 0.2, 0.2]       ← 但实际分配还比较均匀
        #     Σ importance × load = 0.5×0.3 + 0.2×0.3 + 0.2×0.2 + 0.1×0.2 = 0.27
        #                       ← 介于 A 和 B 之间！
        #
        #   你看：内积在均匀时 = 0.25，垄断时 = 1.0。越大越不均匀！
        #
        # 为什么内积有这个性质？
        #   因为 importance 和 load 都是概率分布（每个求和为 1 的非负向量）。
        #   两个概率分布的内积，在它们都是"均匀分布"时取最小值 1/E，
        #   在它们都是"独热分布（one-hot）且指向同一个元素"时取最大值 1。
        #   所以内积天然就是一个"集中度检测器"——
        #   分布越集中在同一个专家上 → 内积越大 → 越需要惩罚。
        #
        #   直观理解：内积就像"问两个人在 E 个问题上是否意见一致"。
        #   → 两个人都说"每个专家都一样好"（= 均匀分布）→ 内积小 → "他们之间没什么共识"
        #   → 两个人都说"只有专家 0 好"（= 独热分布）→ 内积大 → "他们完全一致"
        #   但我们希望的是"缺乏共识"！如果 importance 和 load 高度一致，
        #   说明软偏好和硬分配高度同步——专家 0 既被推崇又被独占，这是坍塌的信号。
        #
        # ─── 最后乘以 E：缩放到合理量级 ───
        # 完美均衡时 Σ = 1/E，乘以 E 后 aux_loss = 1。
        # 完全坍塌时 Σ ≈ 1，乘以 E 后 aux_loss ≈ E（如 4）。
        # 这样 aux_loss 落在 [1, E] 之间，与交叉熵损失的数值量级相近，
        # 不会因为太小（如 0.0625）而淹没在主损失中，也不会因为太大而主导训练。

        # 语法：probs.size(0) 取第 0 维大小 = token 数 S
        #      probs.size(1) 取第 1 维大小 = 专家数 E
        S, E = probs.size(0), probs.size(1)

        # ─── 4a. importance：每个专家的"软受欢迎程度" ───
        #
        # 先看 probs 是什么（它来自前面的 softmax）：
        #   probs 形状 (S, E)，S 个 token，E 个专家。
        #   probs[i, j] = 第 i 个 token 对第 j 个专家的"偏好分数"（已经 softmax 归一化）。
        #   对任意一个 token i，它这一行 probs[i, :] 所有 E 个值加起来 = 1。
        #   你可以把这一行理解为"token i 手里有 1 个单位的关注度，它把这 1 拆成 E 份，
        #   分给 E 个专家，分得多就表示它更信赖那个专家"。
        #
        # 用一个具体例子理解（假设 E=4 个专家，只看 2 个 token）：
        #   token 0: probs[0] = [0.7, 0.2, 0.1, 0.0]  ← token 0 把 70% 的"关注"给了专家 0
        #   token 1: probs[1] = [0.5, 0.3, 0.1, 0.1]  ← token 1 把 50% 的"关注"给了专家 0
        #
        #   现在问题是：在所有 token 眼里，专家 0 到底有多"受欢迎"？
        #   答案：把专家 0 从每个 token 那里分到的"关注"取平均！
        #     importance[0] = (0.7 + 0.5) / 2 = 0.6    ← 平均每个 token 给了专家 0 60% 的关注
        #     importance[1] = (0.2 + 0.3) / 2 = 0.25
        #     importance[2] = (0.1 + 0.1) / 2 = 0.1
        #     importance[3] = (0.0 + 0.1) / 2 = 0.05
        #
        # 代码实现：probs.mean(dim=0)
        #   probs 形状 (S, E)，dim=0 是 token 维度（行方向）。
        #   .mean(dim=0) 沿着 dim=0 求平均 → 把 S 行压缩成 1 行 → 得到长度为 E 的向量。
        #   形状变化：(S, E) → (E,)
        #   语义：importance[j] = 第 j 个专家从每个 token 那里平均分到了多少"关注"。
        #
        # 类比：E 个专家就像 E 个导师，S 个 token 就像 S 个学生。
        #   每个学生有 1 分的心思（=1），他可以按任意比例分配给 E 个导师。
        #   importance[j] = "在所有学生眼里，导师 j 的平均受欢迎评分"。
        #   如果某个导师被所有学生平均认可 → importance 接近 1/E 或更高；
        #   如果某个导师几乎没人关注 → importance 接近 0。
        importance = probs.mean(dim=0)                 # (S,E) → (E,)

        # ─── 4b. load：每个专家的"硬分配负载" ───
        # 取每个 token 的 top-1 专家编号（首选专家，即 softmax 得分最高的那个）。
        # 语法：topk_idx[:, 0] 取所有行、第 0 列（topk 按得分降序排列，第 0 列 = 首选）。
        # 形状：(S, k) → (S,)
        hard1 = topk_idx[:, 0]                         # (S,)

        # ─── 统计每个专家被多少个 token 选为"首选" ───
        # 初始化一个全零的计数器，load 形状 (E,)，E=专家数。
        load = torch.zeros(E, device=x.device)

        # 语法：load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        #
        # 这行代码是"投票开票"的操作。scatter_add_ 的三个参数分别扮演什么角色？
        #   dim=0   → 在哪个维度上做"散射"（这里是 load 的第 0 维，即专家维度）
        #   hard1   → 索引数组，告诉 scatter 每个计数加到哪个位置。形状 (S,)
        #   ones    → 源数组，每个 token 贡献的值。这里全为 1（每 token 投一票）
        #
        # 用具体例子一步步追踪（假设 E=4 个专家，S=6 个 token）：
        #
        #   hard1 = [0, 2, 0, 3, 1, 0]
        #   ↑ token0→专家0, token1→专家2, token2→专家0, token3→专家3,
        #     token4→专家1, token5→专家0
        #
        #   ones = [1, 1, 1, 1, 1, 1]
        #
        #   load 初始: [0, 0, 0, 0]
        #
        #   ！！！scatter_add_ 逐个读取 hard1 中的编号，把 ones 中对应的值"加到"load 的对应位置：
        #
        #     位置 k=0: hard1[0]=0 → load[0] += 1 → load = [1, 0, 0, 0]
        #     位置 k=1: hard1[1]=2 → load[2] += 1 → load = [1, 0, 1, 0]
        #     位置 k=2: hard1[2]=0 → load[0] += 1 → load = [2, 0, 1, 0]
        #     位置 k=3: hard1[3]=3 → load[3] += 1 → load = [2, 0, 1, 1]
        #     位置 k=4: hard1[4]=1 → load[1] += 1 → load = [2, 1, 1, 1]
        #     位置 k=5: hard1[5]=0 → load[0] += 1 → load = [3, 1, 1, 1]
        #
        #   最终 load = [3, 1, 1, 1]，含义：
        #     专家 0 被 3 个 token 选为首选（token 0, 2, 5）
        #     专家 1 被 1 个 token 选为首选（token 4）
        #     专家 2 被 1 个 token 选为首选（token 1）
        #     专家 3 被 1 个 token 选为首选（token 3）
        #
        # 为什么叫 "scatter"（散射）？
        #   hard1 中的值就是"目标地址"，ones 中的值就是"子弹"。
        #   每个子弹被"发射"（scatter）到 load 中 hard1[k] 指定的位置上。
        #   如果多发子弹命中同一位置，它们的效果会累加（add）。
        #
        # 为什么不直接用 bincount？
        #   torch.bincount(hard1) 也能计数，但它只能跑在 CPU 上。
        #   scatter_add_ 在 GPU 上原生支持，效率高得多。
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))

        # 归一化：把绝对计数转为占比（0~1 之间）。
        # 例如 load=[3,1,1,1]，S=6 → load/6 = [0.5, 0.167, 0.167, 0.167]
        # load[j] = 首选专家为 j 的 token 数 / 总 token 数
        # max(S, 1) 防止 S=0 时除零（防御性编程，实际不会发生）
        load = load / max(S, 1)

        # ─── 4c. 辅助损失公式 ───
        # 经过上面的推导，我们已经理解了：Σ(importance × load) 越大，分布越不均匀。
        # 这里直接计算：
        #   aux_loss = E × Σ_j (importance[j] × load[j])
        #
        # 语法：E * (importance * load).sum()
        #   importance * load → 逐元素相乘，形状 (E,) ⊙ (E,) → (E,)
        #   .sum() → 沿所有维度求和，得到一个标量
        #   E * (...) → 缩放因子 E（如 E=4），使范围从 [1/E, 1] 变为 [1, E]
        #
        # 回想一下上面的数值例子（E=4）：
        #   完美均衡：aux_loss = 4 × 0.25 = 1.0（最小）
        #   完全坍塌：aux_loss = 4 × 1.0 = 4.0（最大）
        #   训练时，梯度会推动 aux_loss ↓，即推动分布从"坍塌"走向"均匀"。
        aux_loss = (E * (importance * load).sum())

        # 调试用代码（已注释）：打印中间值，观察路由分布
        print("*"*50)
        print(probs, importance, hard1, load, aux_loss)
        print("*"*50)

        # ─── 返回路由决策 + 辅助损失 ───
        # topk_idx  → 交给 moe.py 的 dispatch 阶段，决定哪些 token 进入哪个专家
        # topk_vals → 交给 moe.py 的 combine 阶段，作为专家输出的加权系数
        # aux_loss  → 累加到总训练损失中，通过反向传播推动门控参数向"均衡路由"方向优化
        return topk_idx, topk_vals, aux_loss

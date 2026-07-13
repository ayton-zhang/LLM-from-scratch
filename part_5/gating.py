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
        # 辅助损失的直觉：
        #   我们希望两样东西都均匀——
        #     importance[i]：专家 i 被所有 token "软偏好"的平均概率（软分配）
        #     load[i]：      专家 i 被选为"首选专家"的 token 占比（硬分配）
        #   如果 importance 和 load 都是均匀分布（每个专家 = 1/E），
        #   那么 importance * load 的和最小 → aux_loss 最小。
        #   如果某个专家垄断了 importance 和 load → 乘积爆炸 → aux_loss 变大 → 惩罚它。

        # 语法：probs.size(0) 取第 0 维大小 = token 数 S
        #      probs.size(1) 取第 1 维大小 = 专家数 E
        S, E = probs.size(0), probs.size(1)

        # ─── 4a. importance：每个专家的"软受欢迎程度" ───
        # probs.mean(dim=0)：沿 token 维度（dim=0）求平均
        #   形状变化：(S, E) → (E,)
        # importance[j] = (1/S) × Σ_i probs[i, j]
        # 含义：在所有 token 中，专家 j 平均分到了多少 softmax 概率质量
        # 如果专家 j 对所有 token 的得分都很低 → importance[j] 接近 0
        # 如果专家 j 平均得分高 → importance[j] 大
        importance = probs.mean(dim=0)                 # (E,)

        # ─── 4b. load：每个专家的"硬分配负载" ───
        # 取每个 token 的 top-1 专家索引（首选专家，即 softmax 得分最高的那个）
        # 语法：topk_idx[:, 0] 取所有行、第 0 列（因为 topk 是按得分降序排列的，
        #       第 0 列就是得分最高的专家）
        # 形状：(S, k) → (S,)
        hard1 = topk_idx[:, 0]                         # (S,)

        # 初始化一个全零的负载计数器，长度 = 专家数
        load = torch.zeros(E, device=x.device)

        # 语法：load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        #   scatter_add_ 是就地操作（带 _ 后缀），沿指定维度做"散射累加"：
        #     第 0 维（dim=0）：按 hard1 中给出的索引来累加
        #     hard1：每个 token 的首选专家编号，形状 (S,)
        #     ones_like：每个 token 贡献计数 1，形状 (S,)
        #   效果：遍历所有 token，对于 token i，执行 load[hard1[i]] += 1
        #   类比：就像投票计数——每个 token 投一票给自己最喜欢的专家，
        #         scatter_add_ 就是开票统计的过程。
        #   为什么不直接用 bincount？
        #     scatter_add_ 直接在 GPU 上高效完成，且与后续操作保持一致的数据类型
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))

        # 归一化：把绝对计数转为占比（fraction of tokens）
        # load[j] = 首选专家为 j 的 token 数 / 总 token 数
        # max(S, 1) 防止 S=0 时除零（防御性编程，实际不会发生）
        load = load / max(S, 1)

        # ─── 4c. 辅助损失公式 ───
        # aux_loss = E × Σ_j (importance[j] × load[j])
        #
        # 为什么是 importance × load？
        #   importance[j]：专家 j 的"平均软权重"（所有 token 对它的平均偏好）
        #   load[j]：      专家 j 的"硬分配占比"（多少 token 把它当首选）
        #   两者相乘再求和，当分布完全均匀时最小（= 1/E × 1/E × E = 1）
        #   当分布极度偏斜时最大（≈ 1 × 1 × E = E，但被 softmax 限制）
        #
        # 乘以 E 的作用：缩放因子，使得均匀分布下的 loss ≈ 1，
        #   与主损失（交叉熵）在数值量级上可比，不会过大或过小
        #
        # 类比：如果把专家比作餐厅窗口——
        #   importance = 每个窗口的平均"关注度"（大家都觉得这家不错）
        #   load = 实际排队人数占比
        #   理想情况：每个窗口的关注度和排队人数都差不多
        #   糟糕情况：一个窗口排长队（load 大）+ 大家还都觉得它好（importance 大）
        #            → 乘积巨大 → aux_loss 惩罚它 → 训练时梯度会把流量推向其他窗口
        aux_loss = (E * (importance * load).sum())

        # 调试用代码（已注释）：打印中间值，观察路由分布
        # print("*"*50)
        # print(probs, importance, hard1, load, aux_loss)
        # print("*"*50)

        # ─── 返回路由决策 + 辅助损失 ───
        # topk_idx  → 交给 moe.py 的 dispatch 阶段，决定哪些 token 进入哪个专家
        # topk_vals → 交给 moe.py 的 combine 阶段，作为专家输出的加权系数
        # aux_loss  → 累加到总训练损失中，通过反向传播推动门控参数向"均衡路由"方向优化
        return topk_idx, topk_vals, aux_loss

# ==========================================
# 奖励模型偏好损失函数集合
# ==========================================
# 职责：提供两种经典的偏好排序损失函数，用于训练奖励模型 (Reward Model, RM)。
#       两种损失函数共享同一个输入接口 (r_pos, r_neg) 但优化目标不同，
#       通过 train_rm.py 的 --loss 参数自由切换。
#
# ─── 什么是偏好损失？───
# 给定同一个 Prompt 的两个回复：
#   - Chosen  (胜出回复):  人类标注者偏好的高质量回复
#   - Rejected (败选回复): 人类标注者认为较差的回复
# 模型分别给它们打分得到 r_pos 和 r_neg（标量，越高表示质量越好）。
# 偏好损失的责任：惩罚模型，当 r_pos <= r_neg 时（即模型"认不出"哪个更好时）给大惩罚，
#                 当 r_pos >> r_neg 时（模型正确判断且信心十足）给低损失。
#
# ─── 两种损失的直观区别 ───
#   Bradley-Terry:  "好坏之间差距越小损失越大，差距足够大时损失趋近 0"（平滑、渐进式优化）
#   Margin Ranking: "差距不到 1.0 就狠狠惩罚，到了就完全不给损失"（硬阈值、快刀斩乱麻式）
#   类比：BT  像是鼓励学生"分数越高越好，没有上限"
#        Margin 像是 "总分及格（60分以上）就行，60 分和 100 分没区别"
# ==========================================

from __future__ import annotations
import torch, torch.nn.functional as F


# ==========================================
# Bradley-Terry 偏好损失（推荐默认）
# ==========================================
def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """-log sigma(r_pos - r_neg) = softplus(-(r_pos - r_neg))

    数学推导：
      Bradley-Terry 模型假设：人类偏好 Chosen 而非 Rejected 的概率为：
        P(chosen > rejected) = sigmoid(r_pos - r_neg)

      用最大似然估计 (MLE) 训练模型，最小化负对数似然：
        loss = -log(P(chosen > rejected)) = -log(sigmoid(r_pos - r_neg))

      恒等变换：-log(sigmoid(x)) = softplus(-x)
      其中 softplus(x) = log(1 + exp(x))，是对 ReLU 的平滑近似。

    直观理解：
      - 当 r_pos >> r_neg 时（模型正确且差距大）：
        sigmoid 接近 1 → -log ≈ 0 → 损失趋近 0（基本不惩罚）
      - 当 r_pos ≈ r_neg 时（模型犹豫不决）：
        sigmoid ≈ 0.5 → -log(0.5) ≈ 0.693 → 模型受到中等惩罚
      - 当 r_pos << r_neg 时（模型判断完全错误）：
        sigmoid 接近 0 → -log → 损失非常大 → 模型受到重罚

    相比 Margin Ranking Loss 的优势：
      - 平滑可导，梯度信号在任何差距下都有意义（不会突然归零）
      - 鼓励模型不断提升评分差距，而非满足于一个固定阈值
      - 这是 RLHF 领域（InstructGPT / LLaMA-2 / Claude 等）最常用的偏好损失

    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html"""
    # ─── 计算得分差 ───
    # r_pos 形状: (B,)，每个元素是 Chosen 回复的标量得分
    # r_neg 形状: (B,)，每个元素是 Rejected 回复的标量得分
    # diff 形状: (B,)，得分差 > 0 → 模型判断正确（Chosen 得分更高）
    #                     < 0 → 模型判断错误（Rejected 得分反而更高）
    diff = r_pos - r_neg

    # ─── 计算损失 ───
    # 语法：F.softplus(x) = log(1 + exp(x))，对输入逐元素计算。
    #   输入为 -diff 的原因：我们希望 diff 越大越好（Chosen 远远高于 Rejected），
    #   softplus(-diff) 在 diff > 0 且很大时接近 0，在 diff <= 0 时随 diff 减小而增大。
    #
    # 形状/数据流动：-diff 形状 (B,) → softplus → 输出 (B,) → .mean() → 标量
    #
    # 为什么用 softplus 而非直接写 -log(sigmoid(diff))？
    #   数值稳定性：当 diff 很大时，sigmoid(diff) → 1.0，
    #   log(1.0) 可能因浮点精度问题丢失，而 softplus(-diff) 直接计算 log(1+exp(-diff))，
    #   在 diff 很大时 exp(-diff)≈0，函数退化为 log(1)=0，数值稳定得多。
    #
    # .mean(): 对 batch 维度求平均，得到标量损失。
    #   设计决策：用 mean 而非 sum——使损失值与 batch_size 无关，
    #   方便不同 batch_size 下的超参（如 lr）复用。
    return F.softplus(-diff).mean()


# ==========================================
# Margin Ranking 边距排序损失（备选方案）
# ==========================================
def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """基于 PyTorch 内置的 MarginRankingLoss 实现边距排序损失。

    数学公式：
      loss = max(0, margin - (r_pos - r_neg))
      = max(0, margin - diff)

    直观理解（以 margin=1.0 为例）：
      - 当 diff >= 1.0 时（Chosen 得分比 Rejected 高出 1.0 以上）：
        margin - diff <= 0 → max(0, 负数) = 0 → 完全不惩罚（"及格了"）
      - 当 diff = 0.5 时（有一些区分但还不够）：
        1.0 - 0.5 = 0.5 → 损失 = 0.5（"还需努力"）
      - 当 diff = -1.5 时（完全判反了）：
        1.0 - (-1.5) = 2.5 → 损失 = 2.5（"严厉惩罚"）

    训练行为特征（与 Bradley-Terry 对比）：
      - 优点：一旦超过 margin 门槛，模型不再被"强迫"拉大差距，
             有时能更专注于分类边界附近（难样本）的优化。
      - 缺点：梯度不连续——在 margin - diff = 0 处梯度突变（见下方导数分析），
             可能导致优化不稳定。
      - 导数分析：∂loss/∂(diff) = 0 (diff >= margin) 或 -1 (diff < margin)
                 Bradley-Terry 的导数 sigmoid(-diff) 在过渡区是平滑的。

    使用场景建议：
      - margin=1.0 是较保守的默认值（要求得分差至少为 1）
      - 调大 margin 要求模型更严格地区分好坏回复（但可能导致欠拟合）
      - 调小 margin 降低要求（但可能导致区分度不足）

    https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html"""

    # ─── 构造目标标签 ───
    # 语法：torch.ones_like(r_pos) 创建一个与 r_pos 形状完全相同、
    #   dtype 和 device 一致的全 1 张量。
    #   形状: (B,)，每个元素都是 1.0
    #   为什么全为 1？F.margin_ranking_loss 的 y 参数：
    #     y=1  → 期望 x1（r_pos）排名高于 x2（r_neg）→ loss = max(0, margin - (x1-x2))
    #     y=-1 → 期望 x2 排名高于 x1         → loss = max(0, margin - (x2-x1))
    #   我们总是期望 r_pos > r_neg，所以 y 全设为 1。
    y = torch.ones_like(r_pos)

    # ─── 调用 PyTorch 内置边距排序损失 ───
    # F.margin_ranking_loss(x1, x2, y, margin=margin) 内部计算：
    #   loss = max(0, -y * (x1 - x2) + margin)
    #   代入 y=1: loss = max(0, margin - (x1 - x2)) = max(0, margin - (r_pos - r_neg))
    #
    # 设计决策：使用 PyTorch 内置函数而非手写 max(0, margin - diff).mean()，
    #   因为内置版本提供了 reduction、更精准的数值处理和 GPU 融合优化。
    #   默认 reduction='mean'，自动对 batch 求平均，返回标量损失。
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)

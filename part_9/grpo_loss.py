# ==========================================
# Part 9 核心损失模块：GRPO (Group Relative Policy Optimization) 策略损失函数
# ==========================================
# 职责：计算 GRPO 算法的核心损失——Clipped Policy Loss（策略剪切损失，与 PPO 相同），
#       外加独立的 KL(π || π_ref) 惩罚项，组合为可进行反向传播的 Total Loss。
#
# ─── GRPO 与 PPO 的关键区别 ───
#   PPO  ：Actor-Critic 结构，需要价值头（Value Head）估计 V(s) 来计算 Advantage
#   GRPO ：纯策略算法（Policy Only），【没有价值头】！
#         优势函数来自"组内相对奖励"：对同一个 Prompt 采样 G 个回答（一组），
#         用该组的奖励均值作为基线（baseline），advantage = 个体奖励 - 组均值。
#         这是 DeepSeekMath 提出的方法，去掉了价值网络，训练更简单、显存更省。
#
# 本文件的实现：
#   total_loss = Policy_Loss - ent_coef * Entropy + kl_coef * KL(π_new || π_ref)
#   ↑ 注意：KL 惩罚是【加到损失上】的（不是改 reward），这是 GRPO 论文的原始做法
# ==========================================

from __future__ import annotations
import torch
from dataclasses import dataclass

# ==========================================
# 损失函数输出数据结构 (Data Container)
# ==========================================
# 语法：@dataclass 自动生成 __init__/__repr__ 等样板方法，
#       比返回元组语义更清晰（out.policy_loss 而非 out[0]）。
@dataclass
class PolicyOnlyLossOut:
    policy_loss: torch.Tensor   # 策略剪切损失标量（Clipped Surrogate Loss）
    entropy: torch.Tensor       # 策略熵估算值标量（未启用时恒为 0）
    approx_kl: torch.Tensor     # 新旧策略间的近似 KL 散度（日志监控用）
    kl_ref: torch.Tensor        # 当前策略与冻结 Reference 模型的 KL 散度（惩罚项）
    total_loss: torch.Tensor    # 最终反向传播总损失 = policy_loss - ent_coef*entropy + kl_coef*kl_ref


# ==========================================
# GRPO 核心损失计算函数 (Policy-Only PPO Loss)
# ==========================================
# 输入都是"动作 token"上的扁平向量（(N_act,)，N_act = 批内所有 response 预测的总数）。
# 这是 GRPO 的优势：没有 value head，loss 只需要 logp 和优势，结构比 PPO 简单。
# ==========================================
def ppo_policy_only_losses(new_logp, old_logp, adv, clip_ratio=0.2, ent_coef=0.0,
                           kl_coef: float = 0.0, kl_mean: torch.Tensor | None = None,
                           token_weights: torch.Tensor | None = None):
    """
    PPO-style clipped policy loss, *policy only* (no value head),
    plus a separate KL(π||π_ref) penalty term:  total = L_PPO + kl_coef * KL.
    Inputs are flat over action tokens: new_logp, old_logp, adv: (N_act,).
    kl_mean is a scalar tensor (possibly weighted to make each response equally important).
    token_weights is optional; when provided, it has shape (N_act,) and is used to
    average token-level policy/entropy/diagnostic terms by response rather than by raw token count.
    """
    # 参数说明：
    #   new_logp    : 更新后策略对动作 token 的对数概率 log π_θ(a|s)，形状 (N_act,)，带梯度
    #   old_logp    : Rollout 采样时旧策略的对数概率 log π_old(a|s)，形状 (N_act,)，常数（已 detach）
    #   adv         : 优势函数标量，形状 (N_act,)——GRPO 中来自"组内相对奖励"并广播到组内每个 token
    #   clip_ratio  : PPO 概率比率剪切范围 ε（默认 0.2，即单步策略更新幅度不超过 ±20%）
    #   ent_coef    : 熵增益系数（GRPO 论文未使用熵奖励，默认 0）
    #   kl_coef     : KL 惩罚系数——对"偏离 Reference 模型"的行为施加的惩罚强度
    #   kl_mean     : KL(π_new || π_ref) 的标量均值（在训练脚本中计算后传入）

    # 空批次保护：若没有动作 token（极端情况），返回全零损失避免除零/空张量运算报错
    if new_logp.numel() == 0:
        # new_logp.new_tensor(0.0) 会继承输入的 device 和 dtype，
        # 比根据 is_cuda 手动决定 device 更兼容 CPU、CUDA 及其他设备。
        zero = new_logp.new_tensor(0.0)
        return PolicyOnlyLossOut(zero, zero, zero, zero, zero)

    # 统一定义加权平均：
    #   没有 weights → 保持普通 token mean，兼容独立调用本函数的旧代码；
    #   有 weights  → 每条 response 的 token 权重之和相同，实现论文中的 response-level mean。
    if token_weights is None:
        def weighted_mean(x):
            return x.mean()
    else:
        weight_sum = token_weights.sum().clamp_min(torch.finfo(new_logp.dtype).eps)

        def weighted_mean(x):
            return (x * token_weights).sum() / weight_sum

    # ─── 1. 策略剪切损失（与 PPO 完全相同的 Clipped Surrogate Objective）───
    # 概率比率：r(θ) = π_new/π_old = exp(log π_new - log π_old)（用对数差算，数值更稳定）
    ratio = torch.exp(new_logp - old_logp)  # (N,)
    # 未截断目标：ratio * adv——好动作（adv>0）增概率、差动作（adv<0）减概率
    unclipped = ratio * adv
    # 截断目标：把 ratio 夹在 [1-ε, 1+ε]，防止单步策略变化过大（PPO 信任域机制）
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # 取两者较小值（悲观下界）并取负均值：最大化目标 → 最小化损失
    # 无论 adv 正负，min 都会选择"更保守"的那个，防止策略单步更新过猛
    policy_loss = -weighted_mean(torch.min(unclipped, clipped))

    # ─── 2. 熵（可选）───
    # 教程简化版：用 -new_logp.mean() 近似熵；鼓励策略保持多样性、防止过早收敛。
    # 注意：ent_coef != 0 时才计算，否则返回 0 张量（GRPO 论文默认不用熵奖励）。
    # 语法：new_logp.new_tensor(0.0) 创建与 new_logp 同设备同类型的 0 标量张量。
    entropy = -weighted_mean(new_logp) if ent_coef != 0.0 else new_logp.new_tensor(0.0)

    # ─── 3. 新旧策略近似 KL（监控用）───
    # 衡量本次更新跑了多远（不参与总损失，纯日志指标）：
    #   过大 → 策略变化太剧烈（学习率可能太大）；接近 0 → 几乎没学到东西
    approx_kl = weighted_mean(old_logp - new_logp)

    # ─── 4. KL(π || π_ref) 惩罚项 ───
    # 由训练脚本算好传入（kl_mean 是标量），衡量当前策略偏离本轮冻结 Reference 的距离。
    # 与 PPO 的区别：PPO 把 KL 惩罚减进 reward（shaped reward），GRPO 把 KL 直接加进 loss。
    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    # ─── 5. 组合总损失 ───
    # total = 策略损失 - 熵奖励 + KL 惩罚
    #   符号解读：
    #   + policy_loss       → 引导策略朝高优势方向更新
    #   - ent_coef*entropy  → 鼓励探索（未启用时为 0，不影响）
    #   + kl_coef*kl_ref    → 惩罚偏离 Reference 太远（防止 Reward Hacking / 语言退化）
    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref # entropy bonus was not used in original GRPO paper
    return PolicyOnlyLossOut(policy_loss, entropy, approx_kl, kl_ref, total)

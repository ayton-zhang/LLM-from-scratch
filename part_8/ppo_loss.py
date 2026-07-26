# ==========================================
# Part 8 核心损失模块：PPO (Proximal Policy Optimization) 剪切损失函数
# 职责：计算 PPO 算法核心的 Policy Clipped Loss（策略剪切损失）、Value Loss（价值均方误差损失）、
#       Entropy Bonus（熵正则化项）以及近似 KL 散度，并组合为可进行反向传播的 Total Loss。
# ==========================================

from __future__ import annotations
import torch, torch.nn.functional as F
from dataclasses import dataclass


# ==========================================
# PPO 损失函数输出数据结构 (Data Container)
# ==========================================
# 语法：@dataclass 是 Python 的数据类装饰器，自动为类生成 __init__、__repr__ 等基础方法，方便强类型返回多项 Loss 指标
@dataclass
class PPOLossOut:
    policy_loss: torch.Tensor  # PPO 策略剪切损失标量
    value_loss: torch.Tensor   # 价值函数 MSE 损失标量
    entropy: torch.Tensor      # 策略熵估算值标量
    approx_kl: torch.Tensor    # 新旧策略间的近似 KL 散度
    total_loss: torch.Tensor   # 最终加权求和的反向传播总损失标量


# ==========================================
# PPO 核心损失计算函数 (PPO Loss Calculation)
# ==========================================
def ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0):
    """计算 PPO 的 Clipped Surrogate Objective 策略损失与 Value 函数损失。

    参数说明:
        new_logp   : 当前更新步骤中，最新 Policy 模型对动作 token 的对数概率 log π_theta(a|s)，形状 (N,)
        old_logp   : Rollout 采样时，Snapshot 旧 Policy 模型对动作 token 的对数概率 log π_old(a|s)，形状 (N,)
        adv        : 归一化后的优势函数 (Advantage Estimation) 标量张量，形状 (N,)
        new_values : 当前更新步骤中，最新 Policy 价值头预测的状态价值 V_theta(s)，形状 (N,)
        old_values : Rollout 采样时预测的旧状态价值 V_old(s)，形状 (N,)
        returns    : 目标的期望回报 Target Value (Returns)，形状 (N,)
        clip_ratio : PPO 概率比率 (Ratio) 的剪切截断范围 epsilon (通常设为 0.1 ~ 0.2)
        vf_coef    : 价值损失在总 Loss 中的加权系数 (Value Head Loss Coefficient)
        ent_coef   : 熵增益系数 (Entropy Bonus Coefficient)
    """

    # ─── 1. 计算 PPO 策略损失 (Clipped Surrogate Objective) ───
    # 数学原理：计算新旧策略的概率比率 ratio = π_theta(a|s) / π_old(a|s)
    # 利用对数性质 log(a) - log(b) = log(a/b)，通过 exp(new_logp - old_logp) 计算比率，数值更加稳定
    # 语法：ratio 形状为 (N,)，其中 N 为动作 token 展平后的总数
    ratio = torch.exp(new_logp - old_logp)  # (N,)

    # 未截断的原始代理目标 (Unclipped Objective)
    unclipped = ratio * adv

    # 截断的代理目标 (Clipped Objective)：强制将 ratio 限制在 [1 - epsilon, 1 + epsilon] 范围内
    # 语法：torch.clamp(..., 1.0 - clip_ratio, 1.0 + clip_ratio) 将张量每个元素限制在区间内，防止单步策略变化过大
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv

    # PPO 核心原理：取 unclipped 与 clipped 的较小值 (torch.min)，并取负均值（因为 PyTorch 优化器做梯度下降，需将最大化目标转为最小化 Loss）
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # ─── 2. 计算价值函数损失 (Value Function Loss) ───
    # 采用均方误差 (MSE) 拟合当前预测价值 new_values 与目标回报 returns
    # 语法：F.mse_loss(new_values, returns) 等价于 mean((new_values - returns)^2)
    value_loss = F.mse_loss(new_values, returns)

    # ─── 3. 计算策略熵 (Entropy Bonus) ───
    # 教程简化版：使用 -new_logp.mean() 作为策略分布熵的极小化近似（严格定义需要全词表分布的期望 -sum(p log p)）
    # 鼓励策略保持多样性，防止过早收敛到局部最优解
    entropy = -new_logp.mean()

    # ─── 4. 估算新旧策略间的近似 KL 散度 (Approximate KL Divergence) ───
    # 用于日志监控：近似计算 KL(π_old || π_new) ≈ E[log π_old - log π_new]
    approx_kl = torch.mean(old_logp - new_logp)

    # ─── 5. 组合最终的反向传播总损失 (Total Loss) ───
    # Total Loss = Policy_Loss + vf_coef * Value_Loss - ent_coef * Entropy
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy

    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)
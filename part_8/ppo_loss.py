# ==========================================
# Part 8 核心损失模块：PPO (Proximal Policy Optimization) 剪切损失函数
# ==========================================
# 职责：计算 PPO 算法核心的 Policy Clipped Loss（策略剪切损失）、Value Loss（价值均方误差损失）、
#       Entropy Bonus（熵正则化项）以及近似 KL 散度，并组合为可进行反向传播的 Total Loss。
#
# ─── PPO 在 RLHF 中的位置与角色 ───
# PPO 是 RLHF（基于人类反馈的强化学习）的第三阶段（也是最后一阶段）训练算法。
# 整个 RLHF 三阶段流水线：
#   Part 6 (SFT)    → 监督微调：让模型学会"按指令格式说话"
#   Part 7 (RM)     → 奖励建模：训练一个"裁判"来评价回答好坏
#   Part 8 (PPO)    → 强化学习微调：用裁判的分数作为奖励信号，优化策略模型
#
# PPO 的核心创新在于"信任域"（Trust Region）机制——它限制了每次策略更新的幅度，
# 防止模型为追求高奖励而"钻空子"（Reward Hacking）或"忘掉"SFT 阶段学到的语言能力。
# 这个限制通过"概率比率剪切"（Probability Ratio Clipping）来实现，是本文件的重点。
#
# ─── PPO 损失的四大组成部分 ───
#   1. Policy Loss（策略损失）   ：最大化被剪切后的优势加权对数概率 → 核心驱动力
#   2. Value Loss（价值损失）    ：让 Critic 更准确地预测未来回报 → 辅助任务
#   3. Entropy Bonus（熵增益）   ：鼓励策略保持探索多样性 → 正则化项
#   4. Approx KL（近似 KL 散度） ：监控新旧策略差异 → 仅用于日志，不参与反向传播
# ==========================================

from __future__ import annotations
import torch, torch.nn.functional as F
from dataclasses import dataclass


# ==========================================
# PPO 损失函数输出数据结构 (Data Container)
# ==========================================
# 为什么用 dataclass 而不是返回元组？
#   1. 语义清晰：访问 out.policy_loss 比 out[0] 更具可读性
#   2. IDE 友好：编辑器能自动补全字段名，不会搞混索引
#   3. 扩展安全：新增字段不影响已有索引访问（元组新增字段会打乱索引顺序）
#
# 语法：@dataclass 是 Python 3.7+ 的数据类装饰器。
#       它自动为类生成 __init__、__repr__、__eq__ 等基础方法，
#       省去了手写 self.policy_loss = policy_loss 的样板代码。
#       与普通类的主要区别：dataclass 用类型注解来声明字段，而非在 __init__ 中赋值。
# ==========================================
@dataclass
class PPOLossOut:
    policy_loss: torch.Tensor  # PPO 策略剪切损失标量（Clipped Surrogate Loss），训练的主要优化目标
    value_loss: torch.Tensor   # 价值函数 MSE 损失标量，衡量 Critic 预测 V(s) 与真实回报的偏差
    entropy: torch.Tensor      # 策略熵估算值标量，值越大表示策略对各种动作的"犹豫程度"越高
    approx_kl: torch.Tensor    # 新旧策略间的近似 KL 散度，用于日志监控策略更新幅度是否过大
    total_loss: torch.Tensor   # 最终加权求和的反向传播总损失标量 = policy_loss + vf_coef * value_loss - ent_coef * entropy


# ==========================================
# PPO 核心损失计算函数 (PPO Loss Calculation)
# ==========================================
# 这是 PPO 算法的"灵魂"所在——整个 Part 8 训练循环都在围绕这个函数运转。
# 每个训练步：Rollout 采样 → 计算 old_logp, adv, returns → 调用 ppo_losses → 反向传播更新参数。
#
# 函数的输入可以分为三组：
#   【概率组】 new_logp, old_logp → 计算策略变化的"幅度"
#   【价值组】 new_values, old_values, returns → 训练 Critic 的预测能力
#   【超参组】 clip_ratio, vf_coef, ent_coef → 控制训练行为的"旋钮"
# ==========================================
def ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0):
    """计算 PPO 的 Clipped Surrogate Objective 策略损失与 Value 函数损失。

    参数说明:
        new_logp   : 当前更新步骤中，最新 Policy 模型对动作 token 的对数概率 log π_θ(a|s)。
                     形状 (N,)，其中 N = 所有 batch 中所有 Response 的 Token 总数（展平后）。
                     这个值带有梯度（requires_grad=True），反向传播会通过它更新 Policy 参数。

        old_logp   : Rollout 采样时，Snapshot 旧 Policy 模型对动作 token 的对数概率 log π_old(a|s)。
                     形状 (N,)。这个值是"冻结的"（从 Rollout 阶段的旧模型计算得到，已 detach），
                     作为 PPO 概率比率的"分母"，不参与反向传播。

        adv        : 归一化后的优势函数 (Advantage Estimation) 标量张量，形状 (N,)。
                     直观含义：当前动作 a_t 比"平均水平"好多少（正 = 好动作，负 = 差动作）。
                     计算方式：Advantage = Returns - V_old(s)，再经过均值-标准差归一化。

        new_values : 当前更新步骤中，最新 Policy 价值头预测的状态价值 V_θ(s)，形状 (N,)。
                     带梯度，Critic 通过它来学习更准确地预测未来累积奖励。

        old_values : Rollout 采样时预测的旧状态价值 V_old(s)，形状 (N,)。
                     已 detach，主要用于监控对比（本函数实际未使用该参数，保留作接口兼容）。

        returns    : 目标的期望回报 Target Value (Returns)，形状 (N,)。
                     计算方式：从最后一个 Token 往前递归计算 R_t = r_t + γ * R_{t+1}（折扣累积奖励）。
                     这是 Critic 要"逼近"的目标值。

        clip_ratio : PPO 概率比率 (Ratio) 的剪切截断范围 ε（epsilon），通常设为 0.1 ~ 0.2。
                     直观含义：允许策略在单次更新中最多把某个动作的概率放大到 1+ε 倍，
                     或缩小到 1-ε 倍。这是 PPO 信任域机制的"硬性约束"。
                     较小的 ε → 更新更保守、训练更稳定但收敛慢；
                     较大的 ε → 更新更激进、收敛快但容易不稳定。

        vf_coef    : 价值损失在总 Loss 中的加权系数（Value Function Coefficient）。
                     通常设为 0.5，让 Policy Loss 主导训练（Policy Loss 权重为 1），
                     Value Loss 作为辅助任务（权重 0.5）。

        ent_coef   : 熵增益系数（Entropy Bonus Coefficient）。
                     通常设为 0.0（关闭）或很小的值如 0.01。
                     设为 0 时表示"不关心策略的多样性"，完全相信 PPO 能找到最优策略；
                     设为正值时鼓励模型在各种动作间保持一定的"犹豫"，防止过早收敛。
    """

    # ==========================================
    # 1. 计算 PPO 策略损失 (Clipped Surrogate Objective)
    # ==========================================
    # 这是 PPO 最核心的创新点。下面逐步拆解其数学直觉和实现细节。
    #
    # ─── 背景：为什么需要"剪切"？───
    # 标准的策略梯度（REINFORCE）直接用 log π(a|s) * Advantage 更新策略，
    # 问题是：如果 Advantage 很大（比如某个回答碰巧得了高分），
    # 模型会在这一步"疯狂"抬高该动作的概率，导致剧烈更新 → 训练不稳定甚至崩溃。
    #
    # PPO 的解决方案：引入"信任域"——限制新旧策略的更新幅度。
    # 具体做法：计算新旧策略的概率比率 r(θ) = π_new(a|s) / π_old(a|s)，
    # 如果 r(θ) 偏离 1 太远（超过 [1-ε, 1+ε]），就把它"剪掉"（clipping），
    # 让这一步的梯度更新"到此为止"，不再进一步拉大策略差距。
    #
    # ─── 步骤 1.1：计算概率比率 (Probability Ratio) ───
    # 数学原理：r(θ) = π_new(a|s) / π_old(a|s)
    # 利用对数恒等式：log(a/b) = log(a) - log(b)
    # 因此 r(θ) = exp(log π_new - log π_old) = exp(new_logp - old_logp)
    #
    # 为什么用 exp(log 差) 而不是直接除概率？
    #   1. 数值稳定：对数概率（log-prob）在 [-∞, 0] 范围内，差值也在这个范围，
    #      而原始概率可能极小（如 1e-9），直接相除可能产生巨大的比率 → 数值溢出。
    #   2. 与 model_logprobs 接口一致：模型返回的就是 log-prob，直接使用避免额外转换。
    #
    # 语法：torch.exp(x) 计算 e^x（自然指数），形状不变 (N,) → (N,)。
    # 直观含义：
    #   ratio ≈ 1  → 新旧策略对这个动作的看法差不多（策略没怎么变）
    #   ratio  > 1  → 新策略比旧策略更"喜欢"这个动作
    #   ratio  < 1  → 新策略比旧策略更"不喜欢"这个动作
    ratio = torch.exp(new_logp - old_logp)  # 形状: (N,)

    # ─── 步骤 1.2：计算两个版本的代理目标 ───
    # 代理目标（Surrogate Objective）的原始形式：ratio * advantage
    # 理解：如果 advantage > 0（好动作），我们希望 ratio 越大越好（增加好动作的概率）；
    #       如果 advantage < 0（差动作），我们希望 ratio 越小越好（减少差动作的概率）。
    #       直接优化这个目标等价于优化"期望优势"。
    unclipped = ratio * adv   # 未截断的原始代理目标，形状: (N,)

    # 截断的代理目标（Clipped Objective）：强制将 ratio 限制在 [1-ε, 1+ε] 范围内。
    # 语法：torch.clamp(ratio, min, max) 逐元素将输入限制在 [min, max] 区间：
    #       小于 min 的值被替换为 min，大于 max 的值被替换为 max，中间的值保持不变。
    #       例如 clip_ratio=0.2 时，ratio 被限制在 [0.8, 1.2] 之间。
    #
    # 直觉：无论 advantage 多大，策略的更新幅度都被"夹死"在 ±20%（ε=0.2）范围内。
    #       这就像给策略更新装了一个"安全带"——再大的奖励波动也不会让模型翻车。
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv  # 形状: (N,)

    # ─── 步骤 1.3：取"悲观"下界（Pessimistic Bound）───
    # 语法：torch.min(unclipped, clipped) 逐元素取两个张量中的较小值。
    #
    # 这是 PPO 论文中最精妙的设计！为什么要取 min？
    #
    # 情况 A：Advantage > 0（这是一个好动作，想要增加它的概率）
    #   → 理想状况下 ratio 应该 > 1，unclipped = ratio * 正数 > adv
    #   → 但 clipped 把 ratio 限制在 1+ε，所以 clipped = (1+ε) * 正数
    #   → torch.min 会选 clipped（较小的那个）= 限制住"想涨多少就涨多少"的冲动
    #
    # 情况 B：Advantage < 0（这是一个差动作，想要减少它的概率）
    #   → 理想状况下 ratio 应该 < 1，unclipped = ratio * 负数（ratio 越小，乘积"负得越少"即越大）
    #   → 但 clipped 把 ratio 限制在 1-ε，所以 clipped = (1-ε) * 负数
    #   → torch.min 会选 clipped（较小的那个）= 限制住"想降多少就降多少"的冲动
    #
    # 总结：无论优势是正还是负，torch.min 都会选择"更保守"的那个目标值，
    # 防止策略在单步更新中变化过大。这就是 PPO"信任域"的精髓。
    #
    # ─── 步骤 1.4：转为损失函数（取负 + 取均值）───
    # 语法：torch.mean(tensor) 对所有元素求平均，返回标量 (0 维张量)。
    # 为什么取负？PyTorch 的优化器（如 Adam）执行梯度**下降**（最小化），
    # 但 PPO 要**最大化**优势加权的对数概率。所以取负号将"最大化"问题转为"最小化"问题。
    # 为什么取均值而非求和？不同 batch 可能有不同数量的 Token (N 不同)，
    # 取均值消除了 batch 规模的影响，使损失在不同 batch 间具有可比性。
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # ==========================================
    # 2. 计算价值函数损失 (Value Function Loss)
    # ==========================================
    # Critic（价值头）的任务：给定状态 s，预测从该状态开始未来能获得的累积奖励 V(s)。
    # Value Loss 衡量 Critic 的预测 V_θ(s) 与"真实值" Returns 之间的差距。
    #
    # 为什么要同时训练 Critic？因为需要它来估算 Advantage（优势函数）：
    #   Advantage(s, a) = Returns - V(s)
    # 如果没有准确的价值估计，Advantage 就会变成噪音，Policy 的优化方向也会跑偏。
    #
    # 语法：F.mse_loss(pred, target) 计算均方误差（Mean Squared Error）：
    #       MSE = (1/N) * Σ(pred_i - target_i)²
    #       等价于 torch.mean((new_values - returns) ** 2)
    # 为什么用 MSE 而非 MAE？MSE 对大误差的惩罚更重（平方关系），
    # 能更快地修正严重偏离的 Critic 预测，这在 RL 训练初期（Critic 预测不准确时）很重要。
    value_loss = F.mse_loss(new_values, returns)

    # ==========================================
    # 3. 计算策略熵 (Entropy Bonus)
    # ==========================================
    # 熵（Entropy）衡量概率分布的"混乱程度"或"不确定性"。
    # 高熵 → 策略对各种动作的偏好比较"平均"（犹豫不决） → 探索性强
    # 低熵 → 策略对某个动作非常"确定"（果断选择） → 容易陷入局部最优
    #
    # 教程简化版实现：使用 -new_logp.mean() 作为熵的粗略近似。
    # 严格定义应该是 H(π) = -Σ π(a|s) * log π(a|s)（对所有动作 a 求和），
    # 但这里我们只有实际采样的动作的对数概率，所以用 -mean(logp) 近似。
    #
    # 直觉：如果模型对每个 Token 都非常确信（log_prob 很高，比如 -0.1），
    # 熵就接近 0，表示策略"太固执"；如果模型犹豫不决（log_prob 很低，比如 -5.0），
    # 熵就很大，表示策略还在"探索"。
    #
    # 注意：熵在总损失中以**负号**出现（total = ... - ent_coef * entropy），
    # 这意味着我们**鼓励**高熵（通过最小化 total loss → 最大化 entropy）。
    # 就像给模型说："别太早下结论，保持好奇心！"
    entropy = -new_logp.mean()

    # ==========================================
    # 4. 估算新旧策略间的近似 KL 散度 (Approximate KL Divergence)
    # ==========================================
    # KL 散度 KL(P||Q) 衡量分布 P 与 Q 的"差异程度"。
    # 这里用 KL(π_old || π_new) ≈ mean(log π_old - log π_new) 的采样近似。
    #
    # 数学：KL(π_old || π_new) = E_{a~π_old}[log π_old(a|s) - log π_new(a|s)]
    #
    # 重点：这个值**不参与反向传播**！它纯粹是日志监控指标，
    # 帮助训练者观察策略在每次更新中的变化幅度。
    #   如果 approx_kl 持续飙高（如 > 0.02）→ 策略变化太快，可能需要调小 learning rate 或 clip_ratio
    #   如果 approx_kl 接近 0         → 策略几乎没变化，可能需要调大 learning rate
    #
    # 注意与 rollout.py 中 approx_kl 的区别：
    #   rollout.py: KL(π_policy || π_ref) — Policy vs Reference 的差距（RLHF 特有的约束）
    #   本文件:    KL(π_old || π_new)   — 新 vs 旧 Policy 的差距（PPO 自身的信任域监控）
    approx_kl = torch.mean(old_logp - new_logp)

    # ==========================================
    # 5. 组合最终的反向传播总损失 (Total Loss)
    # ==========================================
    # Total Loss = Policy_Loss + vf_coef * Value_Loss - ent_coef * Entropy
    #
    # 各项的符号与作用：
    #   + policy_loss    → 引导策略朝"高优势"方向更新（最小化 -mean(min(...)) = 最大化优势）
    #   + vf_coef * value_loss → 次要目标：让 Critic 准确预测价值（辅助 Actor 获得更好的 Advantage 估计）
    #   - ent_coef * entropy    → 正则化：让策略保持一定的"不确定性"，鼓励探索
    #       （注意是减号！最小化 total 等价于最大化 entropy，即鼓励高熵 = 鼓励探索）
    #
    # 这种多任务损失（Multi-Task Loss）的设计在 RL 中非常常见：
    # 一个损失函数同时优化多个相互关联的目标，通过加权系数平衡它们的重要性。
    # 类比：就像学开车时同时优化"到达目的地"（Policy）+"预判路况"（Value）+"保持警觉"（Entropy）。
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy

    # 语法：返回 PPOLossOut 命名元组/数据类，调用方可以通过 .policy_loss、.total_loss 等属性
    #       分别访问各项损失指标，既可以直接取 .total_loss 做反向传播，又可以打印其他指标监控训练。
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)

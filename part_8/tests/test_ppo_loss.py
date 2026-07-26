# ==========================================
# 单元测试：验证 PPO 剪切损失函数 (PPO Loss Objective)
# ==========================================

import torch
from ppo_loss import ppo_losses

def test_clipped_objective_behaves():
    # 设定动作 token 样本数 N=32
    N = 32
    old_logp = torch.zeros(N)                     # 旧 logp = 0
    new_logp = torch.log(torch.full((N,), 1.2))   # 对应 ratio = exp(log(1.2)) = 1.2（超出 1 ± 0.1 截断区间）
    adv      = torch.ones(N)                      # 正优势 adv = 1.0
    new_v = torch.zeros(N)
    old_v = torch.zeros(N)
    ret  = torch.ones(N)                          # Target return = 1.0

    # 调用 ppo_losses 计算，设置 clip_ratio = 0.1
    out = ppo_losses(new_logp, old_logp, adv, new_v, old_v, ret, clip_ratio=0.1)

    # 验证断言：输出的总损失 total_loss 必须为 0 维标量张量
    assert out.total_loss.ndim == 0
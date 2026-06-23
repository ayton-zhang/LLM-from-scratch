import math


# ==========================================
# WarmupCosineLR：训练学习率的“先热身、再降温”调度器
# ==========================================
# 这个类实现的是大模型训练里很常见的学习率策略：
#   1. warmup 阶段：学习率从 0 线性升到 base_lr，避免刚开始训练时步子太大、把参数震飞。
#   2. cosine decay 阶段：学习率按余弦曲线慢慢降到接近 0，让后期更新更细、更稳。
# 它采用 per-step API：训练循环每完成一次 optimizer.step() 后，通常就调用一次 scheduler.step()。
class WarmupCosineLR:
    """Linear warmup → cosine decay (per-step API)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        #   optimizer    : PyTorch 优化器对象，例如 AdamW。
        #                  调度器不直接更新模型参数，而是改 optimizer.param_groups 里的 lr。
        #   warmup_steps : 预热步数。前 warmup_steps 步学习率线性爬坡，
        #                  像汽车刚启动时先慢慢踩油门，减少训练初期的不稳定。
        #   total_steps  : 计划训练的总步数。余弦衰减会根据“当前步 / 总步数”计算进度。
        #   base_lr      : 预热结束时达到的最大学习率，也是余弦衰减的起点。
        self.optimizer = optimizer

        # max(1, warmup_steps) 防止 warmup_steps=0 时后面除以 0。
        # 即使用户传入 0，也至少保留 1 步 warmup，让公式始终安全。
        self.warmup_steps = max(1, warmup_steps)

        # total_steps 至少要比 warmup_steps 多 1。
        # 否则余弦阶段的分母 total_steps - warmup_steps 会变成 0，
        # 就像没有给“降温阶段”留下任何时间。
        self.total_steps = max(self.warmup_steps+1, total_steps)
        self.base_lr = base_lr

        # step_num 记录调度器已经走了多少步。
        # 初始为 0，第一次调用 step() 后变成 1，对应训练的第 1 次参数更新。
        self.step_num = 0

    def step(self):
        # 每调用一次 step()，就向前推进一个训练步。
        # 这个计数是学习率曲线的横坐标：步数越大，学习率走到曲线越靠后的地方。
        self.step_num += 1

        # ─── 阶段 1：线性 warmup ───
        # 在训练刚开始时，模型参数还很“生”，梯度也可能比较剧烈。
        # 因此学习率从 0 按比例逐步升到 base_lr：
        #   第 1 步约为 base_lr / warmup_steps
        #   第 warmup_steps 步正好等于 base_lr
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            # ─── 阶段 2：余弦衰减 ───
            # progress 表示 warmup 结束后，当前已经走过余弦衰减阶段的多少比例：
            #   0.0 附近：刚结束 warmup，学习率接近 base_lr
            #   1.0 附近：接近训练末尾，学习率接近 0
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)

            # 余弦公式：0.5 * base_lr * (1 + cos(pi * progress))
            # 直观理解：cos 从 1 平滑走到 -1，外面的 0.5*(1+...) 把范围压到 [1, 0]。
            # 所以学习率会从 base_lr 平滑降到 0，比直线下降更柔和，后期不容易抖动。
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))

        # optimizer.param_groups 是 PyTorch 的参数组列表。
        # 一个优化器可以有多组参数，例如给 embedding 和 attention 设置不同权重衰减；
        # 这里统一把每个参数组的 lr 更新成同一个调度结果。
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        # 返回当前学习率，方便训练脚本记录日志或打印曲线。
        return lr

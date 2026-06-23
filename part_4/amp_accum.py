import torch


# ==========================================
# AmpGrad：AMP 混合精度 + 梯度累积的训练助手
# ==========================================
# 这个小包装器把训练循环里最容易写乱的两件事收在一起：
#   1. AMP（Automatic Mixed Precision）：前向/反向用更省显存的混合精度训练。
#   2. Gradient Accumulation：多个 micro-batch 的梯度先攒起来，再做一次 optimizer.step()。
#
# 直观理解：
#   - AMP 像是用更轻的工具搬运大部分计算，省显存、提速度；
#   - 梯度累积像是把 4 小杯水倒进同一个量杯，最后按“大 batch”的效果更新一次参数。
class AmpGrad:
    """AMP + gradient accumulation wrapper.
    Usage:
        amp = AmpGrad(optimizer, accum=4, amp=True)
        amp.backward(loss)
        if amp.should_step(): amp.step(); amp.zero_grad()
    """

    def __init__(self, optimizer, accum: int = 1, amp: bool = True):
        #   optimizer : PyTorch 优化器对象，例如 AdamW。
        #               AmpGrad 不直接持有模型，只通过 optimizer 更新它管理的参数。
        #   accum     : 梯度累积步数。accum=4 表示连续 4 个 micro-batch backward，
        #               第 4 次才真正 optimizer.step()，等效 batch size 约放大 4 倍。
        #   amp       : 是否希望启用混合精度。最终还会检查 CUDA 是否可用，
        #               因为这里使用的是 torch.cuda.amp，CPU 上不能按 CUDA AMP 路径运行。
        self.optim = optimizer

        # 至少累积 1 步，避免 accum=0 导致 loss / accum 除以 0。
        # accum=1 时就是普通训练：每个 batch backward 后都可以 step。
        self.accum = max(1, accum)

        # 语法：`amp and torch.cuda.is_available()` 是布尔短路运算。
        # 只有用户要求 AMP 且机器确实有 CUDA GPU 时，self.amp 才为 True；
        # 这样同一份训练代码在 CPU 上也能退化为普通 FP32 训练。
        self.amp = amp and torch.cuda.is_available()

        # GradScaler 是 AMP 的“安全气囊”：
        # FP16 能表示的数字范围较小，小梯度可能下溢成 0。
        # scaler 会先把 loss 放大，再反向传播；真正更新参数前再把梯度缩回真实尺度。
        # enabled=False 时它基本是空操作，方便 CPU/非 AMP 路径共用同一套代码。
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # _n 记录已经执行了多少次 backward。
        # 前面的下划线是 Python 惯例，表示“这是类内部使用的状态，不建议外部直接改”。
        self._n = 0

    def backward(self, loss: torch.Tensor):
        # 梯度累积的关键：每个 micro-batch 的 loss 要除以 accum。
        # 假设 accum=4，如果不除以 4，四次 backward 后梯度会变成原来的 4 倍，
        # 等价于偷偷把学习率放大 4 倍，训练很容易不稳定。
        # 除以 accum 后，累积结果约等于“大 batch 上求平均 loss 再 backward”。
        loss = loss / self.accum

        if self.amp:
            # AMP 路径：先 scale(loss)，再 backward。
            # scale(loss) 返回“放大后的 loss 张量”，它的计算图仍然连着原模型参数；
            # backward() 会把放大后的梯度累积到每个参数的 param.grad 中。
            self.scaler.scale(loss).backward()
        else:
            # 普通 FP32 路径：直接反向传播。
            # PyTorch 的 backward 默认是“累加梯度”而不是覆盖梯度，
            # 所以多次调用 backward 会把多个 micro-batch 的梯度攒到 param.grad 里。
            loss.backward()

        # 每完成一次 backward，就把内部计数加 1。
        # 这个计数决定 should_step() 什么时候返回 True。
        self._n += 1

    def should_step(self):
        # 取模 `%` 用来判断是否刚好累积够 accum 次。
        # 例如 accum=4 时，_n=1/2/3 返回 False，_n=4 返回 True；
        # 然后训练循环才调用 step() 和 zero_grad()，真正更新并清空梯度。
        return (self._n % self.accum) == 0

    def step(self):
        if self.amp:
            # AMP 路径下不能直接 optim.step()：
            # scaler.step(self.optim) 会先检查梯度里是否有 inf/NaN。
            # 如果梯度正常，它会把缩放过的梯度还原，再调用 optimizer.step()；
            # 如果溢出了，它会跳过这次参数更新，避免把模型权重写坏。
            self.scaler.step(self.optim)

            # 根据最近是否发生溢出，动态调整下一轮的缩放因子。
            # 稳定时可能逐渐放大 scale，遇到 inf/NaN 时会降低 scale。
            self.scaler.update()
        else:
            # 非 AMP 路径：梯度已经是真实 FP32 尺度，直接让优化器更新参数。
            self.optim.step()

    def zero_grad(self):
        # 清空梯度要放在真正 step 之后，而不是每个 backward 后。
        # 否则梯度刚累积一部分就被擦掉，grad accumulation 就失效了。
        #
        # set_to_none=True 让 param.grad 变成 None，而不是填充一块全 0 张量。
        # 这样通常更省显存，也能让 PyTorch 在下一次 backward 时重新分配梯度张量。
        self.optim.zero_grad(set_to_none=True)

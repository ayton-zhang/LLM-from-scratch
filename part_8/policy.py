# ==========================================
# Part 8 模块：结合策略语言模型与价值头的 PolicyWithValue 网络
# 职责：将 Part 3 的现代 Transformer 语言模型 (GPTModern) 与极小价值头 (Value Head) 融合，
#       既能输出生成下一个 Token 的 logits（Policy 策略），又能评估当前每个状态的标量价值 Values（Critic 价值函数）。
# ==========================================

from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P

# ─── 跨模块导入 Part 3 的语言模型基础架构 ───
# 语法：sys.path.append(...) 动态添加 part_3 所在路径，实现代码跨章节复用
# 语法：try-except 块提供优雅的降级（Fallback）导入机制，优先兼容自定义路径，若失败则降级为直接导入
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from model_utils.model_modern import GPTModern  # 用户自定义目录结构导入路径
except Exception:
    from model_modern import GPTModern  # 降级备用直接导入路径


# ==========================================
# 策略与价值联合模型 (Actor-Critic Policy Architecture)
# ==========================================
class PolicyWithValue(nn.Module):
    """Policy network = SFT LM + tiny value head.
    NOTE: For simplicity we place value head on top of LM logits (vocab→1).
    This avoids depending on hidden-state internals while keeping the tutorial runnable.
    """
    # ==========================================
    # 初始化方法：构建基础语言模型与价值头
    # ==========================================
    def __init__(self, vocab_size: int, block_size: int, n_layer=4, n_head=4, n_embd=256,
                 use_rmsnorm=True, use_swiglu=True, rope=True, dropout=0.0):
        # 语法：super().__init__() 初始化父类 nn.Module，注册 PyTorch 子模块
        super().__init__()

        # 1. 基础 LM 策略网络（Actor）：加载 Part 3 包含 RMSNorm、SwiGLU 和 RoPE 的现代 GPT 架构
        #   vocab_size  : 词表大小，决定输入嵌入和最终输出 logits 的维度
        #   block_size  : 模型最大上下文窗口长度上限
        #   n_layer     : Transformer 堆叠层数
        #   n_head      : 多头注意力的注意力头数
        #   n_embd      : 隐状态嵌入维度（Hidden dimension）
        #   use_rmsnorm : 是否采用 RMSNorm 归一化（替代标准 LayerNorm，计算更高效）
        #   use_swiglu  : 是否采用 SwiGLU 激活函数 FFN（替代传统 GELU）
        #   rope        : 是否启用 RoPE 旋转位置编码（替代传统可学习位置嵌入）
        #   dropout     : Dropout 随机失活率（评估/推理时设为 0.0）
        self.lm = GPTModern(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd, use_rmsnorm=use_rmsnorm,
                            use_swiglu=use_swiglu, rope=rope, dropout=dropout)

        # 2. 极小价值头（Critic Value Head）：
        # 设计决策：在工业级 PPO 中，Value Head 通常直接连接在 Transformer 最后一层的隐状态 (B, T, n_embd) 上。
        # 本教程为了保持模块独立性、不暴露 GPTModern 内部隐状态接口，巧妙地将 Value Head 建立在 logits (B, T, vocab_size) 之上。
        # 语法：nn.Linear(vocab_size, 1, bias=False) 将 vocab_size 维度的 logits 线性映射为 1 维标量价值。
        self.val_head = nn.Linear(vocab_size, 1, bias=False)

    # ==========================================
    # 前向传播：同时计算对数概率 Logits 与状态价值 Values
    # ==========================================
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # 输入 x 形状: (B, T) —— Batch 大小为 B，序列长度为 T 的 Token ID 张量

        # 1. 调用底层的 GPTModern 进行前向传播：
        #   返回 logits 形状: (B, T, vocab_size) —— 各位置预测词表中每个 Token 的未归一化分值
        #   返回 loss   形状: 标量或 None —— 若传入目标 y 则计算交叉熵损失，否则为 None
        logits, loss, _ = self.lm(x, y)

        # 2. 计算各个时间步的状态价值 (Value Function Estimate)：
        #   self.val_head(logits) 形状: (B, T, vocab_size) -> (B, T, 1)
        #   语法：.squeeze(-1) 消除最后一维，使输出形状重塑为 (B, T)
        #   值的含义：评估从当前 Token 状态开始，未来能够获得的期望折扣回报 (Expected Discounted Return)
        values = self.val_head(logits).squeeze(-1)  # (B, T)

        return logits, values, loss

    # ==========================================
    # 文本生成接口：委托给底层 LM 的自回归生成方法
    # ==========================================
    def generate(self, *args, **kwargs):
        # 语法：*args 和 **kwargs 是 Python 的位置参数与关键字参数打包/解包语法，
        # 将传入的所有参数（如 prompt 张量、max_new_tokens、temperature 等）无缝透传给 self.lm.generate 方法
        return self.lm.generate(*args, **kwargs)
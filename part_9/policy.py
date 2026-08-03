# ==========================================
# Part 9 模块：结合策略语言模型与价值头的 PolicyWithValue 网络
# ==========================================
# 职责：将 Part 3 的现代 Transformer 语言模型 (GPTModern) 与极小价值头 (Value Head) 融合，
#       既能输出生成下一个 Token 的 logits（Policy 策略），又能评估当前每个状态的标量价值 Values（Critic 价值函数）。
#
# 与 Part 8 的关系：本文件与 part_8/policy.py 结构完全一致，是 GRPO 训练管线复用的同一套代码。
# 注意：GRPO（Group Relative Policy Optimization）不需要价值函数（没有 Critic），
#       因此本模块的 Value Head 在 Part 9 中实际上被"忽略"了（train_grpo.py 中只取 logits）。
# ==========================================

from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P
# 尝试优先导入用户自定义目录结构中的 Part 3 语言模型；失败则降级为直接导入
# 语法：sys.path.append(...) 动态将 part_3 目录加入 Python 模块搜索路径，
#       try-except 提供优雅的降级机制，保证代码在不同目录结构下都能运行。
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from model_utils.model_modern import GPTModern  # user-custom path
except Exception:
    from model_modern import GPTModern  # fallback

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
        # 语法：super().__init__() 调用父类 nn.Module 的初始化，注册 PyTorch 子模块
        super().__init__()
        # 基础 LM 策略网络（Actor）：Part 3 的 GPTModern 包含 RMSNorm、SwiGLU、RoPE 等现代架构
        self.lm = GPTModern(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd, use_rmsnorm=use_rmsnorm,
                            use_swiglu=use_swiglu, rope=rope, dropout=dropout)
        # 教学简化版价值头（Critic）：直接建立在 logits 之上而非隐状态之上。
        # 形状变化：(B, T, vocab_size) -> (B, T, 1) -> squeeze -> (B, T)
        # 工业界通常用 nn.Linear(n_embd, 1) 接在隐状态上；这里为了不暴露
        # GPTModern 内部接口，选择在 logits 上做线性映射（bias=False 减少参数）。
        # 注意：GRPO 算法不需要价值头，本模块只是复用了 Part 8 的结构。
        self.val_head = nn.Linear(vocab_size, 1, bias=False)

    # ==========================================
    # 前向传播：同时计算对数概率 Logits 与状态价值 Values
    # ==========================================
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # 调用底层 LM 前向：返回 logits (B,T,V)、loss（有标签时为标量）、_（KV Cache 占位）
        logits, loss, _ = self.lm(x, y)
        # 价值头把每个位置的 vocab 维 logits 映射为 1 个标量，squeeze(-1) 去掉最后维度
        values = self.val_head(logits).squeeze(-1)  # (B,T)
        return logits, values, loss

    # ==========================================
    # 文本生成接口：委托给底层 LM 的自回归生成方法
    # ==========================================
    def generate(self, *args, **kwargs):
        # 语法：*args / **kwargs 将参数（prompt、max_new_tokens、temperature 等）原样透传给底层 LM
        return self.lm.generate(*args, **kwargs)

# ==========================================
# Part 8 辅助模块：Rollout 采样与对数概率/KL散度计算工具集
# 职责：提供通用分词器包装（支持 Part 4 BPE 与 Part 3 ByteTokenizer 双重降级机制）、
#       因果语言模型 Token 对数概率提取、PPO 策略与 Reference 模型之间的 KL 散度估算、
#       以及在线 Rollout 所需的 Prompt 提示词采样。
# ==========================================

from __future__ import annotations
import torch
from typing import List, Tuple

# ─── 跨模块导入与分词器双重降级（Fallback）机制 ───
# 优先加载 Part 4 的 BPETokenizer（高质量、支持加载预训练词表目录）；
# 若导入失败或未训练 BPE，降级为 Part 3 的 ByteTokenizer（基于字节流，无未登录词问题，保证代码可运行）。
import sys
from pathlib import Path as _P

# 1. 尝试导入 Part 4 的 BPE 分词器
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False

# 2. 尝试导入 Part 3 的 ByteTokenizer 作为后备方案
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

# 导入 Part 6 的模板格式化工具（Example 类、prompt 格式化与完整的 (prompt, response) 拼接格式化）
from part_6.formatters import Example, format_example, format_prompt_only


# ==========================================
# 分词器通用封装类 (RLHFTokenizer)
# ==========================================
class RLHFTokenizer:
    """RLHF 统一分词器封装。
    优先尝试加载 BPE 分词器；若失败则自动回退至字节分词器 (ByteTokenizer)。
    """
    # ==========================================
    # 初始化方法：根据可用的分词器模块初始化分词器实例
    # ==========================================
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        # block_size : 上下文最大截断长度
        # bpe_dir    : 预训练 BPE 词表保存路径目录
        # vocab_size : 词表大小（默认 8000）
        self.block_size = block_size
        self.tok = None

        # 第一阶段：尝试初始化并加载 BPE 分词器
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)  # 加载指定目录下的 BPE 词表和 merge 规则
            except Exception:
                self.tok = None

        # 第二阶段：降级回退至 ByteTokenizer
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()

        # 若所有分词器均不可用，抛出运行时异常
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RLHF.")

    # 语法：@property 将方法伪装成只读属性，外部可直接通过 tok.vocab_size 访问
    @property
    def vocab_size(self) -> int:
        # 获取底层分词器的词表大小，若无该属性则默认返回字节分词器的 256
        return getattr(self.tok, 'vocab_size', 256)

    # 文本转 Token ID 列表
    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        # 语法：isinstance(ids, torch.Tensor) 检查返回值是否为 PyTorch 张量，若为张量则转换为 Python 列表
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    # Token ID 列表解码为字符串文本
    def decode(self, ids: List[int]) -> str:
        # 语法：hasattr(self.tok, 'decode') 检查底层分词器是否实现了 decode 方法
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        # 若底层为字节分词器，则将字节数值转化为 bytes 对象再解码为 utf-8 字符串
        return bytes(ids).decode('utf-8', errors='ignore')


# ==========================================
# 对数概率 (Logprob) 工具函数
# ==========================================

# ─── 1. 标签平移（针对因果语言模型的下一个 Token 预测任务）───
def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # 因果语言模型原理：根据前 t 个 token x[:t] 预测第 t+1 个 token x[t+1]。
    # 因此，预测目标标签 labels 需要将序列整体向左平移 1 个单位（即切片去除第 0 个 token）。
    # 输入 x 形状: (B, T) -> 输出形状: (B, T-1)
    # 语法：.contiguous() 确保切片后的张量在内存中连续存储，便于后续张量操作
    return x[:, 1:].contiguous()

# ─── 2. 批量提取目标 Token 的条件对数概率 ───
def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)
    """
    # 输入 logits 形状: (B, T, V) —— 未归一化的分值，V 为词表大小
    # 输入 labels 形状: (B, T) —— 目标 Token 的 ID 张量

    # 1. 计算 Log-Softmax：在词表维度 (dim=-1) 将 logits 转换为对数概率 log p(v | context)
    # 语法：torch.log_softmax 比先算 softmax 再取 log 在数值上更稳定（防止下溢）
    logp = torch.log_softmax(logits, dim=-1)  # 形状: (B, T, V)

    # 2. 从词表维度中精确提取标签 ID 对应的对数概率：
    # 语法：labels.unsqueeze(-1) 将 (B, T) 扩展为 (B, T, 1)，以匹配 logp 的 3 维形状
    # 语法：logp.gather(-1, ...) 在 dim=-1 上按照标签 ID 收集概率值，输出形状 (B, T, 1)
    # 语法：.squeeze(-1) 将最后一维 1 压缩掉，恢复输出形状为 (B, T)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

# ─── 3. 计算语言模型在指定序列上的动作对数概率 (Action Logprobs) ───
@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    # 计算条件概率: log p(x[t+1] | x[:t])
    # 输入 x 形状: (B, T)

    # 1. 前向传播获取模型 logits 分值：
    # 语法：hasattr(model, 'lm') 兼容 PolicyWithValue 封装模型与纯语言模型
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)  # logits 形状: (B, T, V)

    # 2. 对标签进行左移对齐
    labels = shift_labels(x)  # labels 形状: (B, T-1)

    # 3. 截取 logits 前 T-1 个位置与 labels 对齐，计算每个 Token 的条件对数概率
    # logits[:, :-1, :] 形状: (B, T-1, V)
    lp = gather_logprobs(logits[:, :-1, :], labels)
    return lp  # 返回形状: (B, T-1)


# ==========================================
# KL 散度近似计算
# ==========================================

def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    # 针对采样分布的样本，使用 Monte Carlo 估算 KL(π_policy || π_ref)：
    # KL(π||ref) = E_π [log π(a|s) - log ref(a|s)]
    # 输入 policy_logp 与 ref_logp 形状均包含相同数量的动作 token
    # 返回: 所有 token 上对数概率差值的均值标量
    return (policy_logp - ref_logp).mean()


# ==========================================
# Rollout Prompt 提示词数据源采样
# ==========================================
# 尝试导入 HuggingFace datasets 库以加载真实数据集
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None

def sample_prompts(n: int) -> List[str]:
    # 优先方案：从 HuggingFace 的 Alpaca 指令数据集中提取训练 Prompt
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
                if inp:
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            pass  # 若网络不可用或加载失败，跳过并进入降级后备方案

    # 降级后备方案：静态内置的微型示例 Prompt 列表，确保代码无网络连接时也能顺利运行
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    # 语法：(base * ...) 重复拼接列表以满足需求的数量 n，再通过 [:n] 进行精确切片
    return (base * ((n+len(base)-1)//len(base)))[:n]
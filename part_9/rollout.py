# ==========================================
# Part 9 辅助模块：Rollout 采样与对数概率/KL散度计算工具集
# ==========================================
# 职责：提供通用分词器包装（支持 Part 4 BPE 与 Part 3 ByteTokenizer 双重降级机制）、
#       因果语言模型 Token 对数概率提取、策略与 Reference 模型之间的 KL 散度估算、
#       以及 GRPO 训练所需的 Prompt 提示词采样。
#
# 与 Part 8 的关系：本文件与 part_8/rollout.py 功能完全一致，是 GRPO 训练管线复用的工具集。
# ==========================================

from __future__ import annotations
import torch
from typing import List, Tuple

# ─── 跨模块导入与分词器双重降级（Fallback）机制 ───
# 设计动机：优先加载 Part 4 的 BPETokenizer（高质量子词分词）；
# 若导入失败或未训练 BPE，降级为 Part 3 的 ByteTokenizer（基于字节流，永远可运行）。
# 优先级：BPETokenizer > ByteTokenizer
import sys
from pathlib import Path as _P
# 语法：_P(__file__).resolve().parents[1] 是当前文件向上两级目录（项目根目录），拼接 'part_4' 后加入搜索路径
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

# 导入 Part 6 的模板格式化工具（Example 类、prompt 格式化与完整的 (prompt, response) 拼接格式化）
from part_6.formatters import Example, format_example, format_prompt_only

# ---------- tokenizer helpers ----------
# ==========================================
# 分词器通用封装类 (RLHFTokenizer)
# ==========================================
# 设计动机：GRPO 训练中需要一致的分词/解码接口，但底层可能是 BPE 或 ByteTokenizer。
# 本类统一封装，对外暴露 encode() / decode() / vocab_size 三个统一接口。
class RLHFTokenizer:
    # 初始化方法：根据可用的分词器模块初始化分词器实例
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        # block_size : 上下文最大截断长度（Transformer 注意力机制的硬性限制）
        # bpe_dir    : 预训练 BPE 词表保存路径目录（None 则使用随机初始化词表，不推荐）
        # vocab_size : BPE 词表大小，默认 8000（对 ByteTokenizer 无影响，其词表固定 256）
        self.block_size = block_size
        self.tok = None
        # 第一阶段：尝试初始化并加载 BPE 分词器
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)  # 从磁盘加载已训练的 merge 规则和词表
            except Exception:
                self.tok = None
        # 第二阶段：降级回退至 ByteTokenizer（字节级兜底方案，不依赖预训练词表）
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        # 最终检查：所有分词器均不可用则直接报错，避免后续出现难以排查的 AttributeError
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RLHF.")

    # 语法：@property 将方法伪装成只读属性，外部可直接访问 tok.vocab_size 而无需调用
    @property
    def vocab_size(self) -> int:
        # 语法：getattr(obj, 'attr', default) 安全读取属性，不存在时返回默认值 256（ByteTokenizer 词表大小）
        return getattr(self.tok, 'vocab_size', 256)

    # 文本 → Token ID 列表
    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        # 语法：isinstance(ids, torch.Tensor) 统一处理"返回张量 or 返回列表"两种分词器实现
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    # Token ID 列表 → 文本
    def decode(self, ids: List[int]) -> str:
        # 语法：hasattr(obj, 'decode') 检测底层分词器是否实现 decode 方法
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        # 字节分词器路径：字节 ID 序列 → bytes → UTF-8 字符串，errors='ignore' 跳过无法解码的字节
        return bytes(ids).decode('utf-8', errors='ignore')

# ---------- logprob utilities ----------
# ==========================================
# 对数概率 (Logprob) 工具函数
# ==========================================

# ─── 1. 标签平移（因果 LM 的下一个 Token 预测任务）───
def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # 因果语言模型原理：给定 x[0..t] 预测 x[t+1]，因此标签需要整体左移一位。
    # 输入 (B, T) -> 输出 (B, T-1)；.contiguous() 确保切片后内存连续，便于后续张量操作。
    return x[:, 1:].contiguous()

# ─── 2. 批量提取目标 Token 的条件对数概率 ───
def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)
    """
    # log_softmax 在词表维 (dim=-1) 上做归一化，比先 softmax 再 log 数值更稳定（防 exp 溢出）
    logp = torch.log_softmax(logits, dim=-1)  # (B, T, V)
    # 语法拆解：labels.unsqueeze(-1) 把 (B,T) 扩为 (B,T,1) 与 logp 的 3 维对齐；
    #          logp.gather(-1, ...) 沿词表维按标签 ID 收集对应对数概率（像从抽屉里取指定编号的东西）；
    #          .squeeze(-1) 去掉最后一维的 1，恢复 (B,T)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

# ─── 3. 计算语言模型在指定序列上的动作对数概率 ───
@torch.no_grad()
# 语法：@torch.no_grad() 关闭自动求导——Rollout 阶段只需要前向推理值，
#       不构建计算图可大幅节省显存和计算时间。
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    # 计算条件概率 log p(x[t+1] | x[:t])，输入 x 形状 (B, T)
    # 语法：hasattr(model, 'lm') 兼容 PolicyWithValue 封装（取 .lm）与纯语言模型两种形式；
    #       第二个参数 None 表示不使用 KV Cache（一次性处理完整序列）。
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)
    labels = shift_labels(x)      # (B, T-1)，标签左移
    lp = gather_logprobs(logits[:, :-1, :], labels)  # logits 舍尾（预测者舍尾），与 labels 对齐
    return lp  # (B, T-1)

# ---------- KL ----------
# ==========================================
# KL 散度近似计算
# ==========================================
def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
              weights: torch.Tensor | None = None) -> torch.Tensor:
    # GRPO 直接加到 loss 的 KL 使用论文中的 k3 估计器，而不是简单的 log-ratio：
    #   k3 = exp(log π_ref - log π_policy)
    #        - (log π_ref - log π_policy) - 1
    # 令 x = log π_ref - log π_policy，则 exp(x) - x - 1 >= 0，
    # 因此这个估计值不会像简单的 (policy_logp - ref_logp) 那样出现负数。
    # 这里仍然只用采样到的动作 token，而不是遍历完整词表，所以它是 Monte Carlo 估计。
    log_ratio = ref_logp - policy_logp
    kl = torch.exp(log_ratio) - log_ratio - 1.0

    # weights 用来实现论文中的“每条 response 等权”聚合：
    #   每个 token 的权重 = 1 / 该 response 的 token 数；
    #   这样一条回答内部先平均，再对回答平均，长回答不会因为 token 更多而占更大权重。
    if weights is None:
        return kl.mean()
    weight_sum = weights.sum().clamp_min(torch.finfo(kl.dtype).eps)
    return (kl * weights).sum() / weight_sum

# ---------- small prompt source ----------
# ==========================================
# Rollout Prompt 提示词数据源采样
# ==========================================
# 尝试导入 HuggingFace datasets 库；导入失败则降级为本地静态列表（确保无网络也能运行）
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None

def sample_prompts(n: int) -> List[str]:
    # 优先方案：从 Alpaca 指令数据集（前 24 条）提取 instruction + input 作为 Prompt
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
                if inp:                    # 有 input 字段则拼接到指令后面（用换行分隔）
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:          # 收集够 n 条提前退出
                    break
            if arr:
                return arr
        except Exception:
            pass                           # 网络/加载失败则静默降级到后备方案
    # 降级后备方案：内置微型示例 Prompt 列表，保证离线环境也能运行
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    # 语法：(base * m)[:n] 重复拼接列表 m 次再精确切片；m 用向上取整除除法保证够 n 条
    return (base * ((n+len(base)-1)//len(base)))[:n]

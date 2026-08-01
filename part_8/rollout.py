# ==========================================
# Part 8 辅助模块：Rollout 采样与对数概率/KL散度计算工具集
# ==========================================
# 职责：提供通用分词器包装（支持 Part 4 BPE 与 Part 3 ByteTokenizer 双重降级机制）、
#       因果语言模型 Token 对数概率提取、PPO 策略与 Reference 模型之间的 KL 散度估算、
#       以及在线 Rollout 所需的 Prompt 提示词采样。
#
# 在 PPO（Proximal Policy Optimization）训练流程中，本模块所处的位置：
#   Rollout 阶段：
#     1. sample_prompts()  → 获取一批 Prompt 文本
#     2. RLHFTokenizer     → 将 Prompt 编码为 Token ID
#     3. Policy 模型生成   → 自回归生成 Response（实际 rollout 在 PPO trainer 中完成）
#     4. model_logprobs()  → 计算 Policy 模型对生成序列的对数概率 log π(a|s)
#     5. model_logprobs()  → 计算 Reference 模型对同一序列的对数概率 log π_ref(a|s)
#     6. approx_kl()       → 用两者的差值估算 KL 散度，作为 PPO 的约束项
#
# 这些工具函数贯穿 PPO 的 采样→打分→约束 三个环节，是 RLHF 训练的基础设施。
# ==========================================

from __future__ import annotations
import torch
from typing import List, Tuple

# ─── 跨模块导入与分词器双重降级（Fallback）机制 ───
# 设计动机：本模块需要兼容两种分词器，但用户可能只安装了其中一种，甚至两种都没训练好。
# 因此采用"尝试导入 → 失败则降级"的策略，确保代码在任何环境下都能运行（至少 ByteTokenizer 不依赖外部词表）。
#
# 优先级：Part 4 的 BPETokenizer（高质量子词分词） > Part 3 的 ByteTokenizer（字节级兜底方案）
import sys
from pathlib import Path as _P

# 1. 尝试导入 Part 4 的 BPE 分词器
# 语法：_P(__file__).resolve().parents[1] 表示当前文件向上两级目录（即项目根目录），
#       再拼接 'part_4' 即得到 part_4 目录的绝对路径，追加到 sys.path 后可以 import 其中的模块。
#       .parents 是一个序列，parents[0] = 当前目录, parents[1] = 父目录, 以此类推。
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True       # 全局标志位：标记 BPE 分词器是否可用，后续 RLHFTokenizer 据此决定初始化策略
except Exception:
    _HAS_BPE = False

# 2. 尝试导入 Part 3 的 ByteTokenizer 作为后备方案
# ByteTokenizer 直接将每个字节当作一个 Token，词表固定为 256，不需要训练，永不失败。
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None   # 如果连 ByteTokenizer 都导入不了（极少见），置为 None

# 导入 Part 6 的模板格式化工具（Example 类、prompt 格式化与完整的 (prompt, response) 拼接格式化）
# 这些工具用于将原始文本组装成模型可直接消费的"对话模板"格式
from part_6.formatters import Example, format_example, format_prompt_only


# ==========================================
# 分词器通用封装类 (RLHFTokenizer)
# ==========================================
# 设计动机：PPO 训练中需要一致的分词/解码接口，但底层分词器可能是 BPE 或 ByteTokenizer。
# 本类对两种分词器做了统一封装，对外暴露 encode() / decode() / vocab_size 三个统一接口，
# 内部自动处理类型转换（如 torch.Tensor → list）和能力检测（如是否有 decode 方法）。
#
# 类比：就像 USB-C 转接头——不管插的是 BPE 还是 ByteTokenizer，上层代码都用同一个接口。
# ==========================================
class RLHFTokenizer:
    """RLHF 统一分词器封装。
    优先尝试加载 BPE 分词器；若失败则自动回退至字节分词器 (ByteTokenizer)。
    """
    # ==========================================
    # 初始化方法：根据可用的分词器模块初始化分词器实例
    # ==========================================
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        # ─── 参数说明 ───
        #   block_size : 上下文最大截断长度。超过此长度的输入将被截断，
        #                这是 Transformer 模型注意力机制的硬性限制（由训练时的位置编码大小决定）。
        #   bpe_dir    : 预训练 BPE 词表保存路径目录。若提供，将从该目录加载已训练的 BPE 词表
        #                和 merge 规则文件；若为 None，则使用随机初始化的词表（不推荐）。
        #   vocab_size : BPE 词表大小，默认 8000。仅在初始化 BPETokenizer 时传入，
        #                控制子词合并操作的目标词表容量。对 ByteTokenizer 无影响（它的词表固定为 256）。
        self.block_size = block_size
        self.tok = None    # 底层分词器实例，初始为 None，后续通过两阶段尝试来赋值

        # ─── 第一阶段：尝试初始化并加载 BPE 分词器 ───
        # BPE（Byte Pair Encoding）是本项目的"主力"分词器，子词粒度介于字符和单词之间，
        # 兼具语义表达能力与泛化性（遇到生僻词时自动拆解为已知子词组合）。
        if _HAS_BPE:
            try:
                # 用指定 vocab_size 初始化一个空的 BPE 分词器对象
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    # 若提供了预训练目录，从该目录加载 .model 和 .vocab 文件
                    # 这样分词器才知道具体的合并规则和词表映射
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None   # 加载失败时重置为 None，准备进入第二阶段降级

        # ─── 第二阶段：降级回退至 ByteTokenizer ───
        # 当 BPE 分词器不可用（未训练 / 导入失败 / 加载出错）时，自动切换到字节分词器。
        # ByteTokenizer 将文本的每个字节映射为 0~255 的整数 ID，不依赖任何预训练词表，
        # 是最"硬核"但永远可用的降级方案。
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()

        # ─── 最终检查：若所有分词器均不可用，直接报错 ───
        # 这种情况极少（ByteTokenizer 也导入失败），但不能默默吞掉错误，
        # 否则后续 encode/decode 调用会出现难以排查的 AttributeError。
        if self.tok is None:
            raise RuntimeError("No tokenizer available for RLHF.")

    # ─── vocab_size 属性 ───
    # 语法：@property 将方法伪装成只读属性，外部可直接通过 tok.vocab_size 访问（无需加括号调用）。
    #       这是一种 Python 惯用法，让接口更简洁：用户写 `tok.vocab_size` 而非 `tok.vocab_size()`。
    @property
    def vocab_size(self) -> int:
        # 获取底层分词器的词表大小。
        # 语法：getattr(obj, 'attr', default) 尝试读取 obj.attr，若属性不存在则返回 default 值。
        #       这里用 getattr 而非直接 self.tok.vocab_size，是防止底层分词器未定义该属性导致崩溃。
        #       ByteTokenizer 的词表固定为 256（0x00~0xFF 共 256 个字节），所以默认值设为 256。
        return getattr(self.tok, 'vocab_size', 256)

    # ─── 编码：文本 → Token ID 列表 ───
    def encode(self, text: str) -> List[int]:
        """将自然语言文本转换为 Token ID 整数序列。"""
        ids = self.tok.encode(text)
        # 语法：isinstance(ids, torch.Tensor) 检查返回值是否为 PyTorch 张量。
        #       BPETokenizer.encode() 可能返回 torch.Tensor 或 Python list（取决于实现），
        #       这里统一转换为 Python 列表，保证接口一致性。
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    # ─── 解码：Token ID 列表 → 文本 ───
    def decode(self, ids: List[int]) -> str:
        """将 Token ID 整数序列还原为人类可读的文本字符串。"""
        # 语法：hasattr(obj, 'name') 检查对象是否拥有某个属性/方法。
        #       这里先检查底层分词器是否实现了 decode 方法（BPE 分词器有），
        #       若没有则走字节分词器的解码路径。
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        # 字节分词器的解码路径：将整数 ID（0~255）依次转为 bytes 对象，再用 UTF-8 解码为字符串。
        # errors='ignore' 表示遇到无法解码的字节序列时直接跳过，而非抛出 UnicodeDecodeError。
        # 这在处理部分损坏的 Token 序列时很关键——宁可少输出几个字符，也不要让整个 rollout 崩溃。
        return bytes(ids).decode('utf-8', errors='ignore')


# ==========================================
# 对数概率 (Logprob) 工具函数
# ==========================================
# 在 PPO 训练的 Rollout 阶段，我们需要精确计算模型对"已生成序列"的条件对数概率
# log π(a_t | s_t)，其中 a_t 是第 t 步选择的 Token，s_t 是前 t 个 Token 构成的上下文。
#
# 这三个函数是层层递进的关系：
#   shift_labels()       → 处理因果 LM 的预测-标签对齐（纯数学操作）
#   gather_logprobs()    → 从 logits 中提取目标 Token 的对数概率（概率提取）
#   model_logprobs()     → 完整的前向 + 对齐 + 提取流程（端到端接口）
# ==========================================

# ─── 1. 标签平移（针对因果语言模型的下一个 Token 预测任务）───
def shift_labels(x: torch.Tensor) -> torch.Tensor:
    """将输入序列右移一位，使预测与标签对齐。

    因果语言模型（Causal LM）的训练目标：给定 x[0], x[1], ..., x[t]，预测 x[t+1]。
    换句话说：模型在位置 t 输出的 logits，其"正确答案"是位置 t+1 的实际 Token。

    因此，我们需要把原始序列 x[:, 1:] 作为标签（丢弃第 0 个位置的标签，因为没有"前文"能预测它）。

    直观理解（假设 seq_len=5）：
        输入序列:    [A,  B,  C,  D,  E]
        模型预测:    [B', C', D', E', ? ]    ← 每个位置预测"下一个" Token
        对齐后标签:  [B,  C,  D,  E]        ← 去掉第一个，与预测对齐

    输入形状: (B, T)  →  输出形状: (B, T-1)
    其中 B = batch_size, T = 序列总长度
    """
    # 语法：x[:, 1:] 是二维切片，取所有 batch、从第 1 个位置到末尾（跳过第 0 个）。
    #       .contiguous() 确保切片后的张量在内存中连续存储。
    #       切片操作（如 [:, 1:]）返回的是原始张量的"视图"（view），内存可能不连续，
    #       后续某些 PyTorch 操作（如 .view()）要求连续内存，这里预先把内存整理好，避免后续报错。
    return x[:, 1:].contiguous()

# ─── 2. 批量提取目标 Token 的条件对数概率 ───
def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs of the given labels.
    logits: (B,T,V), labels: (B,T) over same T
    returns: (B,T) log p(labels)
    """
    # ─── 形状说明 ───
    # 输入 logits 形状: (B, T, V) —— 模型输出的未归一化分值，V = vocab_size（词表大小）
    #   三维含义：B 个样本，每个样本有 T 个位置，每个位置输出 V 个候选词的"原始分数"
    # 输入 labels 形状: (B, T) —— 目标 Token 的整数 ID 张量，每个值范围 [0, V-1]
    # 输出形状:          (B, T) —— 每个位置对应的对数概率 log p(label_token | context)

    # ─── 步骤 1：Logits → Log-Probabilities ───
    # 类比：logits 是每位考生的"卷面原始分"，log_softmax 把原始分归一化为"对数概率"——
    #       分数最高的考生概率最大，但所有考生概率之和 = 1（在概率空间），取 log 后求和 ≠ 1。
    logp = torch.log_softmax(logits, dim=-1)  # 形状: (B, T, V) —— 每个位置、每个候选词的对数概率

    # ─── 步骤 2：从 V 个候选词中精确提取 labels 指定的那个 Token 的对数概率 ───
    # 语法拆解（三步走）：
    #   ① labels.unsqueeze(-1)：将 labels 从 (B, T) 扩展为 (B, T, 1)，
    #      新增的最后一维用于匹配 logp 的第三维（V 维），作为 gather 操作的"索引维"。
    #   ② logp.gather(-1, ...)：在 dim=-1（词表维）上，按照 labels 中的 Token ID 收集对应的对数概率。
    #      输出形状为 (B, T, 1) —— 最后一个维度被压缩为 1（只取了一个抽屉）。
    #   ③ .squeeze(-1)：将最后一维的 1 挤掉，恢复为 (B, T)。
    #      等于把"装在一个盒子里的单个值"直接拿出来，形状更简洁。
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

# ─── 3. 计算语言模型在指定序列上的动作对数概率 (Action Logprobs) ───
@torch.no_grad()
# 语法：@torch.no_grad() 是 PyTorch 的上下文管理器装饰器。
#       被装饰的函数内部所有张量操作都不会构建计算图、不追踪梯度。
#       在这里使用是因为：Rollout 阶段的 logprob 计算只需要前向推理值，
#       不需要反向传播，关闭 autograd 能大幅节省显存和计算时间。
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    """计算因果语言模型对序列 x 的条件对数概率：log p(x[t+1] | x[:t])。

    输入 x 形状: (B, T) —— 一个完整的 Token ID 序列（prompt + response）
    输出形状:    (B, T-1) —— 每个位置（除最后一个）对"下一个 Token"的对数概率

    用途：PPO 训练中需要分别计算 Policy 模型和 Reference 模型对同一段生成文本
          的对数概率，两者差值用于估算 KL 散度（防止 Policy 偏离 Reference 太远）。
    """
    # ─── 步骤 1：模型前向传播，获取 logits ───
    # 语法：hasattr(model, 'lm') 用于兼容两种模型封装形式：
    #   - PolicyWithValue 封装模型：model.lm 是底层的语言模型（用于算 logprob），
    #     model 本身还有一个 value head（用于算状态价值）。
    #   - 纯语言模型：直接 model(x, None) 调用即可。
    # 无论哪种形式，第二个参数 None 表示不使用 KV Cache（Rollout 阶段通常一次性处理完整序列）。
    # 返回值解包：
    #   logits: (B, T, V) — 我们需要的预测分值
    #   _     : 占位符，忽略 loss 返回值（推理时 loss 为 None 或不需要）
    #   _     : 占位符，忽略 KV Cache 返回值（Rollout 不需要缓存）
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)

    # ─── 步骤 2：标签对齐（预测 vs 目标错位 1 步）───
    # 因果 LM 的核心对齐逻辑：
    #   logits[t] 是对"看到 x[:t] 后预测 x[t+1]"的分数
    #   labels[t] = x[t+1] 是该预测的正确答案
    # 因此 labels 需要右移一位（去掉第 0 个位置的标签）。
    labels = shift_labels(x)  # labels 形状: (B, T-1)，丢弃了原始序列的第 0 个 Token 作为标签

    # ─── 步骤 3：从 logits 中提取对应 labels 的对数概率 ───
    # logits[:, :-1, :]：取前 T-1 个位置的预测（丢弃最后一个位置的预测，因为没有 labels[T] 与之对应）
    # gather_logprobs()：在这些预测中提取 labels 指定的 Token 的对数概率
    # 最终得到形状 (B, T-1)：序列中每个"下一步预测"的对数概率。
    lp = gather_logprobs(logits[:, :-1, :], labels)
    return lp


# ==========================================
# KL 散度近似计算
# ==========================================
# KL 散度（Kullback-Leibler Divergence）衡量两个概率分布之间的"距离"（严格说不是距离，因为不对称）。
# 在 PPO 训练中，我们用它来约束 Policy 模型不要偏离 Reference 模型太远：
#   如果 Policy 的对数概率和 Reference 的对数概率相差太大 → KL 惩罚项增大 → 限制策略更新幅度。
#
# 数学定义（离散分布）：
#   KL(π_policy || π_ref) = Σ π_policy(a|s) * [log π_policy(a|s) - log π_ref(a|s)]
#
# 但我们没有"所有可能动作的完整分布"（动作空间是整个词表 V，遍历每个 Token 计算不现实）。
# 因此用 Monte Carlo 估算：从 Policy 实际采样出的动作来近似期望——
#   KL ≈ (1/N) * Σ [log π_policy(a_i|s_i) - log π_ref(a_i|s_i)]
# 其中 (a_i, s_i) 是从 Policy 的实际生成轨迹中采样的（状态-动作对）。
#
# 这个近似的直觉：我们不用"穷举所有可能 Token 的概率"，而是用"实际选中 Token 的概率差"
# 来代表整个分布的偏移程度。采样越充分（N 越大），估算越准。
# ==========================================

def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Monte Carlo 近似估算 KL 散度：KL(π_policy || π_ref)。

    输入：
        policy_logp: 形状 (B, T-1) 或一维 —— Policy 模型对序列中每个 Token 的对数概率 log π(a|s)
        ref_logp:    同上形状 —— Reference（冻结）模型对同一序列的对数概率 log π_ref(a|s)
    输出：一个标量（0 维张量），表示整批数据上的平均 KL 散度近似值。

    直观理解：如果 policy_logp ≈ ref_logp，说明 Policy 模型和 Reference 模型想法一致，
              KL → 0，PPO 不会施加惩罚；如果 policy_logp 远大于 ref_logp（Policy 对某些
              Token 过于自信），KL 增大，PPO 的 trust-region 机制会限制本次更新幅度。
    """
    # 语法：.mean() 对所有元素求平均。
    #       policy_logp - ref_logp 是逐元素相减（广播语义），结果形状与输入相同。
    #       为什么用均值而非求和？因为不同序列长度不同，用均值可以消除长度影响，
    #       使 KL 估算在不同 batch 间具有可比性。
    return (policy_logp - ref_logp).mean()


# ==========================================
# Rollout Prompt 提示词数据源采样
# ==========================================
# 在线 Rollout 训练需要一批 Prompt 作为"起点"，让 Policy 模型根据每个 Prompt 生成 Response。
# 本函数以"优先联网 → 降级本地"的策略获取 Prompt，确保在各种环境下都能运行。
#
# 最佳实践：使用真实的指令数据集（如 Alpaca）能提供多样化的 Prompt，避免模型过拟合到
# 几个固定的"罐头"问题。本地静态列表只是兜底方案。
# ==========================================

# 尝试导入 HuggingFace datasets 库以加载真实数据集
# datasets 库是 HuggingFace 生态的核心组件，提供统一的 Dataset 加载接口。
# 但它不是 PyTorch 的依赖，用户可能没安装，所以用 try/except 做优雅降级。
try:
    from datasets import load_dataset as _load_ds
except Exception:
    _load_ds = None   # 导入失败时置 None，后续通过 if _load_ds is not None 判断

def sample_prompts(n: int) -> List[str]:
    """采样 n 个 Prompt 文本，优先从 Alpaca 数据集获取，失败则用本地静态列表。

    参数 n: 需要的 Prompt 数量。在 PPO 训练中，通常等于 batch_size。
            例如 batch_size=8，则每轮 rollout 采集 8 个不同的 Prompt。

    返回: 长度为 n 的字符串列表，每个字符串是一个完整的 Prompt（如指令或问题）。
    """
    # ─── 优先方案：从 HuggingFace 的 Alpaca 指令数据集中提取训练 Prompt ───
    # Alpaca 是 Stanford 发布的指令微调数据集，包含 52K 条 (instruction, input, output) 三元组。
    # 这里只提取 instruction + input 作为 Prompt（不包含 output），让模型自己生成答案。
    # split="train[:24]" 只取训练集的前 24 条，原因：
    #   1. 演示/实验场景下不需要太多数据
    #   2. 减少网络下载量和加载时间
    #   3. 前 24 条足够覆盖多样化的指令类型
    if _load_ds is not None:
        try:
            ds = _load_ds("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                # 提取 instruction 字段（核心指令），去除首尾空白
                inst = (r.get('instruction') or '').strip()
                # 提取 input 字段（可选的补充输入，如"请翻译：Hello world"），去除首尾空白
                inp = (r.get('input') or '').strip()
                # 若存在 input 字段，将其追加到 instruction 后面，用换行分隔
                # 例如 instruction="翻译" + input="Hello world" → "翻译\nHello world"
                if inp:
                    inst = inst + "\n" + inp
                # 过滤掉空的 Prompt（instruction 和 input 都为空的情况）
                if inst:
                    arr.append(inst)
                # 收集到足够数量后提前退出循环，避免不必要的遍历
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            # 可能的失败原因：网络不可用、磁盘空间不足、数据集格式变更等。
            # 这里静默跳过（pass），不打印错误信息，保持训练日志整洁，
            # 因为降级方案能提供完全相同功能的替代。
            pass

    # ─── 降级后备方案：静态内置的微型示例 Prompt 列表 ───
    # 当 HuggingFace datasets 未安装、网络不可用或数据集加载失败时，
    # 使用硬编码的示例 Prompt 作为兜底。这些 Prompt 涵盖了常见的 LLM 任务类型：
    #   解释概念、列举优缺点、总结知识点、编写代码。
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    # ─── 语法拆解：如何用 4 条 Prompt 满足任意数量 n 的需求 ───
    # (base * m) 表示将 base 列表重复拼接 m 次，例如 base * 2 = [...4条..., ...4条...] = 8条。
    # m = (n + len(base) - 1) // len(base) 是"向上取整除法"，计算需要重复多少次。
    #   例如 n=3:  (3+4-1)//4 = 6//4 = 1  → 重复 1 次，取前 3 条
    #   例如 n=5:  (5+4-1)//4 = 8//4 = 2  → 重复 2 次，取前 5 条
    #   例如 n=10: (10+4-1)//4 = 13//4 = 3 → 重复 3 次，取前 10 条
    # 这保证了无论 n 是多少都能返回恰好 n 条 Prompt，不会出现 IndexError。
    return (base * ((n+len(base)-1)//len(base)))[:n]

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# 依赖包导入与环境兼容性处理
# ==========================================
# 语法：使用 try-except 动态尝试导入 Hugging Face 的 datasets 库。
# 这样做的好处是“优雅降级”（Graceful Degradation）：
# 如果环境未安装 `datasets` 包，代码不会抛出异常终止，
# 而是将 load_dataset 标为 None，后续逻辑会自动切换到本地备用数据集。
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# ==========================================
# 数据结构：偏好数据样例容器 (PrefExample)
# ==========================================
# 语法：@dataclass 装饰器用于自动为类生成 __init__、__repr__ 等基础方法。
# 作为一个高效的数据容器，避免手写繁琐的构造函数样板代码。
#
# 深度学习概念 - 偏好数据对 (Preference Pair)：
# 在 RLHF（基于人类反馈的强化学习）和 RM（奖励模型）训练中，
# 数据集的核心形态是一对比较数据：包含同一个 Prompt 的两条回答，
# - chosen: 人类标注员认为更好的回答（Positive Example / 选优项）
# - rejected: 人类标注员认为较差的回答（Negative Example / 淘汰项）
@dataclass
class PrefExample:
    prompt: str      # 提示词/问题输入（部分数据集如 Anthropic HH 已将 prompt 混入对话全文，此处可留空）
    chosen: str      # 优质回答（人类偏好）
    rejected: str    # 劣质回答（人类拒绝）


# ==========================================
# 数据加载入口：加载偏好数据集 (load_preferences)
# ==========================================
def load_preferences(split: str = "train[:200]") -> List[PrefExample]:
    """Load a tiny preference set. Tries Anthropic HH; falls back to a toy set.
    HH fields: 'chosen', 'rejected' (full conversations). We use an empty prompt.
    """
    # 初始化输出容器列表，用于存放解析好的 PrefExample 对象
    items: List[PrefExample] = []

    # 第一优先级：尝试在线加载真实的开源偏好数据集 Anthropic/hh-rlhf
    if load_dataset is not None:
        try:
            # 语法：split="train[:200]" 是 datasets 库的切片语法，表示只加载训练集的前 200 条数据。
            # 这对于本地微型实验、Demo 运行和快速 Sanity Check 非常有用，无需下载庞大的完整数据。
            ds = load_dataset("Anthropic/hh-rlhf", split=split)
            for row in ds:
                # 语法：row.get("key", "") 安全获取字典字段，若不存在则返回空字符串；
                # .strip() 去除文本首尾多余的空格和换行符。
                ch = str(row.get("chosen", "")).strip()
                rj = str(row.get("rejected", "")).strip()

                # 只有当 chosen 和 rejected 均非空时，才视为有效偏好数据
                if ch and rj:
                    # Anthropic HH 数据集的 chosen/rejected 包含了完整对话上下文，因此 prompt 传空字符串即可
                    items.append(PrefExample(prompt="", chosen=ch, rejected=rj))
        except Exception:
            # 如果因为网络问题、超时或 Hugging Face 连接失败，打印提示信息并自动降级
            print("Failed to load Anthropic/hh-rlhf dataset. Using fallback toy pairs.")
            pass

    # 第二优先级：离线备用数据（Fallback Toy Pairs）
    # 如果未安装 datasets 库或网络下载失败（items 为空），使用本地预设的测试样例。
    # 这确保了代码在离线、无网环境下也可以 100% 可运行、可测试！
    if not items:
        # fallback toy pairs
        items = [
            PrefExample("Summarize: Scaling laws for neural language models.",
                        "Scaling laws describe how performance improves predictably as model size, data, and compute increase.",
                        "Scaling laws are when you scale pictures to look bigger."),
            PrefExample("Give two uses of attention in transformers.",
                        "It lets the model focus on relevant tokens and enables parallel context integration across positions.",
                        "It remembers all past words exactly without any computation."),
        ]
    return items
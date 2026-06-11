# ==========================================
# 工具函数：top-k / top-p（核采样）过滤
# ==========================================
# 在文本生成中，模型输出的 logits 经过 softmax 后是一个 vocab_size 维的
# 概率分布。如果直接从这个分布采样，低概率的"噪音" token 可能会被选中，
# 导致生成质量下降（出现乱码、重复、不合逻辑的内容）。
#
# 本函数提供两种经典的"截断采样"策略，只保留高质量的候选 token：
#
#   top-k 过滤：只保留概率最高的 k 个 token，其余置为 -inf（概率归零）。
#               例：vocab=50000, top_k=50 → 只从 top-50 中采样。
#
#   top-p 过滤（核采样 / Nucleus Sampling）：
#               按概率从高到低累加，只保留累积概率 ≤ p 的那些 token。
#               例：top_p=0.9，保留概率最高的若干个 token 直到它们累积概率
#               达到 0.9，其余置为 -inf。
#               相比 top-k，top-p 能动态调整候选数量——分布尖锐时选得少，
#               分布平坦时选得多，更灵活智能。
#
# 两种过滤可以同时使用（先 top-k 缩小范围，再 top-p 动态截断），
# 也可以单独使用（只传其中一个参数）。
#
# 使用方式（model_modern.py 的 generate 方法中）：
#   next_logits = top_k_top_p_filtering(next_logits / temperature, top_k=50, top_p=0.9)
#   probs = torch.softmax(next_logits, dim=-1)  # 过滤后再 softmax
from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - logits: (B, vocab)
    Returns filtered logits with -inf for masked entries.
    """
    # 输入 logits 形状 (B, vocab_size)，B 通常为 1（生成时 batch=1）。
    # 语法：`B, V = logits.shape` 元组解包，同时获取 batch 大小和词表大小。
    B, V = logits.shape

    # clone() 创建一份拷贝，避免修改原始 logits。
    # 虽然这里调用方不依赖原始值，但函数式风格（不修改输入）是良好实践。
    filtered = logits.clone()

    # ==========================================
    # 第一步：top-k 过滤
    # ==========================================
    # 只保留 logits 最高的 top_k 个 token，其余设为 -inf。
    # -inf 经过 softmax 后概率为 0，确保这些 token 永远不会被选中。
    #
    # 条件 `top_k < V`：如果词表只有 256 个 token 而 top_k=300，
    # 那 top-k 过滤没有意义（全部候选都保留），直接跳过。
    if top_k is not None and top_k < V:
        # torch.topk(filtered, top_k, dim=-1)：
        #   沿最后一维（vocab_size）找 top_k 个最大的 logits。
        #   返回具名元组 (values, indices)，这里只需要 values，用 _ 忽略 indices。
        #   topk_vals 形状 (B, top_k)，如 top_k=50 则每行 50 个值。
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)

        # 取第 k 大的值作为阈值：(B, top_k) → [:, -1] → (B,) → unsqueeze(-1) → (B, 1)。
        # 语法：[:, -1] 取每行最后一个（最小的 top-k 值，即第 k 大的值）。
        # .unsqueeze(-1) 在最后加一维：(B,) → (B, 1)，这样广播比较时能对齐 (B, V)。
        kth = topk_vals[:, -1].unsqueeze(-1)

        # 语法：filtered[filtered < kth] = float('-inf') 是布尔索引赋值。
        # filtered < kth 返回 (B, V) 的布尔张量，True 的位置表示"不在 top-k 内"。
        # 把这些位置设为 -inf（极大负值），softmax 后概率变为 0。
        filtered[filtered < kth] = float('-inf')

    # ==========================================
    # 第二步：top-p（核采样 / Nucleus Sampling）过滤
    # ==========================================
    # 核心思路：按概率从高到低累加，找到"累积概率刚好超过 top_p"的那个位置，
    # 该位置之后的低概率 token 全部设为 -inf。
    #
    # 步骤拆解：
    #   1. 对 logits 降序排序
    #   2. 计算 softmax 概率
    #   3. 计算累积概率（cumsum）
    #   4. 标记累积概率超过 top_p 的位置
    #   5. 把这些位置设为 -inf
    #   6. 把排序后的结果"散列"回原始顺序
    #
    # 条件 `0 < top_p < 1.0`：top_p=0 意味着不保留任何 token（无意义），
    # top_p=1 意味着保留全部（等于不过滤），这两种情况都跳过。
    # None 表示不启用 top-p 过滤。
    if top_p is not None and 0 < top_p < 1.0:
        # ─── 2.1 降序排序 ───
        # torch.sort(filtered, descending=True, dim=-1)：
        #   沿词表维降序排列 logits。返回 (sorted_values, sorted_indices)。
        #   sorted_idx 记录每个位置"原来在词表的哪个位置"，
        #   用于最后把排序后的结果映射回原始词表顺序。
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)

        # ─── 2.2 计算累积概率 ───
        # softmax 把排序后的 logits 转为概率分布，形状 (B, V)。
        probs = torch.softmax(sorted_logits, dim=-1)

        # torch.cumsum(probs, dim=-1)：沿最后一维计算累积和。
        # cumsum[i] = probs[0] + probs[1] + ... + probs[i]。
        # 因为已降序排列，cumsum 是从高到低累加，快速增长。
        cumsum = torch.cumsum(probs, dim=-1)

        # ─── 2.3 标记需要被过滤的位置 ───
        # cumsum > top_p：累积概率超过阈值的位置标记为 True（需要过滤）。
        # 例：top_p=0.9，cumsum = [0.3, 0.55, 0.75, 0.92, 0.98, ...]
        #     cumsum > 0.9 → [F, F, F, T, T, ...]，位置 3 起被过滤。
        mask = cumsum > top_p

        # 至少保留 1 个 token（概率最高的那个永远不被过滤）。
        # 语法：mask[..., 0] = False 把每行第 0 个（概率最高的）强制设为 False。
        # 即使 top_p 极小（如 0.01），也保证至少有一个候选。
        # keep at least 1 token
        mask[..., 0] = False

        # 把被标记的位置设为 -inf（概率归零）。
        sorted_logits[mask] = float('-inf')

        # ─── 2.4 散列回原始词表顺序 ───
        # sorted_logits 现在是按降序排列的，但调用方期望按原始词表顺序。
        # 需要用 sorted_idx 把排序后的值"映射回"原始位置。
        #
        # 先创建一张全是 -inf 的空白表，形状 (B, V)。
        # 语法：torch.full_like(tensor, value) 创建与 tensor 同形状的张量，
        #       所有元素填充为 value。比 zeros_like + fill 更简洁。
        # Scatter back
        filtered = torch.full_like(filtered, float('-inf'))

        # scatter_(dim, index, src)：沿指定维度，按 index 把 src 的值"撒"到目标位置。
        # 语义：filtered[i][index[i][j]] = src[i][j]（对每行 i 的每列 j）。
        # 这里把 sorted_logits（排序后，部分为 -inf）按 sorted_idx（原始位置）
        # 散列回 filtered，恢复原始词表顺序。
        #
        # 语法：scatter_ 带下划线后缀表示 in-place 操作（原地修改张量），
        # 不创建新张量，节省内存。
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered

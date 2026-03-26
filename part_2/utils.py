from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - logits: (B, vocab)
    Returns filtered logits with -inf for masked entries.
    """
    # logits.shape 返回张量的形状元组，例如 (4, 50257)。
    # 用 B, V 分别接收批次大小和词表大小。
    B, V = logits.shape

    # .clone() 深拷贝张量，避免修改原始 logits（防止影响调用方的数据）。
    filtered = logits.clone()

    # ── top-k 过滤 ──────────────────────────────────────────────
    # top_k is not None：确保调用时传入了 top_k 参数。
    # top_k < V：如果 top_k >= 词表大小，过滤等于没做，直接跳过。
    if top_k is not None and top_k < V:
        # torch.topk 返回每行最大的 top_k 个值及其索引。
        # dim=-1 表示在最后一维（vocab 维）上操作。
        # _ 忽略索引，只要值。topk_vals 形状为 (B, top_k)。
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)

        # topk_vals[:, -1] 取每行第 top_k 个（最小的那个）值，形状 (B,)。
        # .unsqueeze(-1) 在最后加一维，变为 (B, 1)，方便后面做广播比较。
        kth = topk_vals[:, -1].unsqueeze(-1)

        # 布尔索引：将所有小于第 k 个值的位置直接设为负无穷。
        # 经过 softmax 后，-inf 对应的概率为 0，这些 token 不会被采样到。
        filtered[filtered < kth] = float('-inf')

    # ── top-p（nucleus）过滤 ─────────────────────────────────────
    # 0 < top_p < 1.0：top_p 必须是有效的概率值才执行。
    if top_p is not None and 0 < top_p < 1.0:
        # 将 logits 按从大到小排序。
        # torch.sort 返回 (排序后的值, 原始索引)。
        # descending=True 表示降序（高分在前）。
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)

        # 对排好序的 logits 做 softmax，得到概率分布。
        probs = torch.softmax(sorted_logits, dim=-1)

        # torch.cumsum 计算累积和。例如 [0.4, 0.3, 0.2, 0.1] → [0.4, 0.7, 0.9, 1.0]。
        # 这样可以找到"概率累计达到 top_p"的边界位置。
        cumsum = torch.cumsum(probs, dim=-1)

        # cumsum > top_p 得到一个布尔 mask：累计概率超过 top_p 之后的位置为 True，
        # 这些位置对应的 token 排名靠后，需要被过滤掉。
        mask = cumsum > top_p

        # 强制保留排名第一的 token（概率最高的那个），防止所有 token 都被过滤。
        # mask[..., 0] = False 将每行第 0 个位置（最高分 token）的 mask 设为 False（不过滤）。
        mask[..., 0] = False

        # 将需要过滤的位置（mask 为 True）的 logits 设为 -inf。
        sorted_logits[mask] = float('-inf')

        # 此时 sorted_logits 是按排序顺序排列的，需要还原回原始 vocab 顺序。
        # torch.full_like 创建一个与 filtered 形状相同、全部填充 -inf 的张量。
        filtered = torch.full_like(filtered, float('-inf'))

        # .scatter_(dim, index, src)：按 index 把 src 的值写回到 filtered 对应位置。
        # sorted_idx 记录了排序前每个元素的原始位置，用它把值"散射"回去，完成逆排序。
        filtered.scatter_(1, sorted_idx, sorted_logits)

    # 返回过滤后的 logits：不在 top-k/top-p 范围内的位置已被设为 -inf。
    return filtered
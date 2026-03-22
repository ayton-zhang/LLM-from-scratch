import torch  # 导入 PyTorch 库，用于张量操作

def causal_mask(T: int, device=None):
    """Returns a bool mask where True means *masked* (disallowed).
    Shape: (1, 1, T, T) suitable for broadcasting with (B, heads, T, T).
    """
    # torch.ones((T, T), dtype=torch.bool, device=device)：
    # 创建一个形状为 (T, T) 的全 True 张量，数据类型为 bool，放在指定设备上
    # torch.triu(..., diagonal=1)：
    # 取上三角矩阵，diagonal=1 表示从对角线上方开始（不包括对角线），
    # 上三角为 True，下三角为 False
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    # m.view(1, 1, T, T)：重新变形张量为 (1, 1, T, T)，添加两个维度 1，便于广播
    return m.view(1, 1, T, T)
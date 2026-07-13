# ==========================================
# Part 5 MoE 演示脚本：Mixture-of-Experts 端到端跑通 + 路由可视化
# ==========================================
#
# 这个脚本用一个随机张量模拟一批 token，喂入 MoE 层，观察：
#   1. 前向传播是否正常运行（输出形状是否正确、aux loss 是否有值）
#   2. 路由分布——每个专家被多少个 token 选中（直方图）
#
# 为什么路由分布重要？
#   MoE 的核心效率来自"稀疏激活"——每个 token 只走少数专家。
#   但如果路由总是选同一两个专家，其他专家就"闲置"了，模型容量被浪费。
#   直方图能让我们一眼看出是否存在这种"专家坍塌"问题。
#
# 运行方式：
#   cd part_5
#   python demo_moe.py                                          # 默认参数
#   python demo_moe.py --tokens 12 --hidden 256 --experts 8 --top_k 2

# ==========================================
# 导入区
# ==========================================

# argparse：构建命令行接口，让用户灵活调整 MoE 参数
import argparse
# torch：PyTorch 核心库
import torch
# 从 moe 模块导入 MoE 类（本 Part 的核心组件）
from moe import MoE


# ==========================================
# 主入口：参数解析 → 构建数据 → 前向传播 → 路由分析
# ==========================================
if __name__ == "__main__":
    # ─── 步骤 1：解析命令行参数 ───
    # 每个参数都允许用户在命令行覆盖，方便快速实验不同配置
    p = argparse.ArgumentParser()

    # --tokens：输入序列的总 token 数（batch 内所有序列的 token 总和）
    #   type=int  → 自动把命令行字符串转为整数
    #   default=64 → 不传参数时默认 64 个 token
    p.add_argument('--tokens', type=int, default=64)

    # --hidden：隐藏维度（每个 token 的特征向量长度）
    #   也是每个专家 MLP 的输入/输出维度
    p.add_argument('--hidden', type=int, default=128)

    # --experts：专家总数
    #   4 是研究中最常见的起步配置；生产模型（如 Mixtral 8×7B）用 8 个专家
    p.add_argument('--experts', type=int, default=4)

    # --top_k：每个 token 选中的专家数
    #   1 = 最稀疏（每个 token 只走 1 个专家），2 = 常见配置（增加容错）
    p.add_argument('--top_k', type=int, default=1)

    # --cpu：强制使用 CPU 运行，即使有 GPU
    #   action='store_true' → 标志型参数，传了为 True，不传默认 False
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    # ─── 步骤 2：选择计算设备 ───
    # 语法：torch.device('字符串') 创建一个设备对象
    # 语法：`A if 条件 else B` 三元表达式：
    #   有 CUDA 且没指定 --cpu → 用 GPU（'cuda'）
    #   否则 → 用 CPU
    # GPU 上 MoE 的 dispatch/combine 操作涉及大量索引散射，GPU 并行优势明显
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ─── 步骤 3：构造模拟输入 ───
    # torch.randn 生成标准正态分布 N(0,1) 的随机张量，模拟一批 token 的隐状态
    # 形状设计：(B=2, T=tokens//2, C=hidden)
    #   B=2（batch size）  → 两条独立的"句子"
    #   T=tokens//2        → 每条句子的长度 = 总 token 数 ÷ 2
    #   C=hidden           → 每个 token 的特征维度
    # 例如 --tokens 64 --hidden 128 → 形状 (2, 32, 128)
    x = torch.randn(2, args.tokens//2, args.hidden, device=device)  # (B=2,T=tokens/2,C)

    # ─── 步骤 4：构建 MoE 层 ───
    # MoE(dim=..., n_expert=..., k=...) 创建一个稀疏 MoE 层：
    #   dim       → token 的隐藏维度（输入/输出维度）
    #   n_expert  → 专家总数（每个专家是一个独立的小 MLP）
    #   k         → top_k，每个 token 激活的专家数
    # .to(device) 把模型参数搬到 GPU（如果 device='cuda'）
    moe = MoE(dim=args.hidden, n_expert=args.experts, k=args.top_k).to(device)

    # ─── 步骤 5：前向传播 ───
    # 语法：torch.no_grad() 上下文管理器
    #   作用：禁用梯度计算，节省显存并加速（这里只是演示，不需要反向传播）
    #   类比：就像考试时只"答题"不"复盘"，省去了记录每一步推导过程的开销
    with torch.no_grad():
        # 语法：`y, aux = moe(x)` 元组解包
        #   moe.forward() 返回两个值：
        #     y   → MoE 层的输出，形状与输入 x 相同 (B, T, C)
        #     aux → 辅助损失（auxiliary loss），用于负载均衡训练，标量
        y, aux = moe(x)

    # ─── 步骤 6：路由分布直方图 ───
    # 目的：看看每个专家被多少个 token 选中，判断负载是否均衡
    from gating import TopKGate

    # 拿到 MoE 层的门控模块（路由器）
    gate = moe.gate

    # 语法：x.view(-1, args.hidden)
    #   把 (B, T, C) → (B*T, C)，把所有 batch 和序列维展平为一个大 batch
    #   -1 告诉 PyTorch "这一维的大小你帮我算"
    # gate() 返回 (indices, weights, aux_loss)：
    #   idx      → 形状 (N_tokens, k)，每个 token 被选中的 k 个专家索引（整数）
    #   w        → 形状 (N_tokens, k)，对应专家的门控权重（浮点数，softmax 归一化后）
    #   _        → 辅助损失（这里不需要，用 _ 占位丢弃）
    # 语法：`idx, w, _ = gate(...)` 中 _ 是 Python 惯例，表示"这个返回值我不需要"
    idx, w, _ = gate(x.view(-1, args.hidden))

    # 语法：idx[:, 0]
    #   取每个 token 的"首选专家"（top_k 中权重最大的那个），形状 (N_tokens,)
    # 语法：torch.bincount(tensor, minlength=N)
    #   统计张量中每个非负整数的出现次数，返回长度为 minlength 的张量
    #   例如 bincount([0,2,2,1,0], minlength=4) → [2,1,2,0]（0出现2次,1出现1次,2出现2次,3出现0次）
    #   minlength 确保直方图长度 = 专家数，即使某专家从未被选中也会显示 0
    hist = torch.bincount(idx[:, 0], minlength=args.experts)

    # ─── 步骤 7：输出结果 ───
    # tuple(y.shape)：把 torch.Size 转为普通元组，打印更美观
    # float(aux)：aux 是 0 维张量（标量张量），float() 转为 Python 浮点数
    # .4f 格式化：保留 4 位小数
    print(f"Output shape: {tuple(y.shape)} | aux={float(aux):.4f}")

    # 打印路由直方图——每个专家的"工作量"一目了然
    # 理想情况：各专家的 counts 大致均匀（如 [8,8,8,8] 表示 32 个 token 均匀分配）
    # 糟糕情况：某个专家 counts=0 或某个专家独占绝大多数 token
    #   → 说明门控坍塌，需要用 aux loss 或负载均衡策略矫正
    print("Primary expert load (counts):", hist.tolist())

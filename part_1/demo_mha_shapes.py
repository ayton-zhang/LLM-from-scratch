"""Walkthrough of multi-head attention with explicit matrix math and shapes.
Generates a text log at ./out/mha_shapes.txt.
"""  # 原有说明不变，提示这是步骤演示程序
import os  # 系统路径和文件操作模块
import math  # 数学模块，用来开平方
import torch  # PyTorch 张量和运算库
from multi_head import MultiHeadSelfAttention  # 取自同目录的多头自注意力类

OUT_TXT = os.path.join(os.path.dirname(__file__), 'out', 'mha_shapes.txt')  
# 拼接路径：当前脚本所在目录下的 'out' 子目录中的 'mha_shapes.txt' 文件，
# 用于记录多头注意力形状变化的日志


def log(s):
    print(s)  
    # 把信息 s 打印到控制台屏幕上，让用户能实时看到每一步的输出
    with open(OUT_TXT, 'a') as f:  
        # 以追加模式（'a'）打开日志文件 OUT_TXT，这样不会覆盖之前的内容，而是继续往后写
        f.write(s + "\n")  
        # 把信息 s 写进文件 f 中，并在末尾加换行符 \n，确保每条日志占一行


if __name__ == "__main__":
    # Reset file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)  
    # 创建输出目录，如果不存在的话；exist_ok=True 避免目录已存在时报错
    open(OUT_TXT, 'w').close() 
    # 以写模式打开日志文件（会清空原有内容），然后立即关闭，相当于新建或重置文件

    B, T, d_model, n_head = 1, 5, 12, 3  # 设定批次、序列长度、模型维度、头数
    d_head = d_model // n_head  # 每个头的维度
    x = torch.randn(B, T, d_model)  # 随机输入 (B,T,d_model)
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)  # 实例化注意力模块

    log(f"Input x:           {tuple(x.shape)} = (B,T,d_model)")
    qkv = attn.qkv(x)  # 线性层得到 QKV 叠在一起，(B,T,3*d_model)
    log(f"Linear qkv(x):     {tuple(qkv.shape)} = (B,T,3*d_model)")

    qkv = qkv.view(B, T, 3, n_head, d_head)  # 变形为 (B,T,3,heads,d_head)
    log(f"view to 5D:        {tuple(qkv.shape)} = (B,T,3,heads,d_head)")

    q, k, v = qkv.unbind(dim=2)  # 拆成 q,k,v 三个张量，每个维度是 (B,T,heads,d_head)
    log(f"q,k,v split:       q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    # 把 head 维调到第二维，方便每个头独立计算注意力，(B,heads,T,d_head)
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)  
    v = v.transpose(1, 2)
    log(f"transpose heads:   q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} = (B,heads,T,d_head)")

    scale = 1.0 / math.sqrt(d_head)  # 缩放因子，防止 scores 太大
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # 计算 q@k^T，结果 (B,heads,T,T)
    log(f"scores q@k^T:      {tuple(scores.shape)} = (B,heads,T,T)")

    weights = torch.softmax(scores, dim=-1)  # 把 scores 归一化为权重
    log(f"softmax(weights):  {tuple(weights.shape)} = (B,heads,T,T)")

    ctx = torch.matmul(weights, v)  # 用权重加权 v -> 上下文 (B,heads,T,d_head)
    log(f"context @v:        {tuple(ctx.shape)} = (B,heads,T,d_head)")

    # ctx 当前形状 (B,heads,T,d_head)，需要变回 (B,T,d_model) 以输出给后续层
    out = ctx.transpose(1, 2)  
    # transpose(1,2) 交换第1和第2维：(B,heads,T,d_head) -> (B,T,heads,d_head)
    out = out.contiguous()  
    # contiguous() 确保张量在内存中连续存储，方便后续 view 操作
    out = out.view(B, T, d_model)  
    # view() 重新变形为 (B,T,d_model)，把多个头拼回一个大维度
    log(f"merge heads:       {tuple(out.shape)} = (B,T,d_model)")

    out = attn.proj(out)  # 最终投影
    log(f"final proj:        {tuple(out.shape)} = (B,T,d_model)")

    log("\nLegend:")
    log("  B=batch, T=sequence length, d_model=embedding size, heads=n_head, d_head=d_model/heads")
    log("  qkv(x) is a single Linear producing [Q|K|V]; we reshape then split into q,k,v")
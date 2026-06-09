from __future__ import annotations
import torch
import torch.nn as nn
from block_modern import TransformerBlockModern
from tokenizer import ByteTokenizer

# ==========================================
# 路径修复：让 Python 能找到兄弟目录下的模块
# ==========================================
# os.path.dirname(__file__) 获取本文件 (model_modern.py) 所在的目录，即 part_3/
# os.path.join(..., '..') 向上走一级，指向项目根目录 llm_from_scratch/
# os.path.abspath(...) 把路径转换为绝对路径，确保无论从哪里运行脚本都能正确找到模块。
# sys.path.insert(0, ...) 把根目录插入模块搜索路径的最前面，这样 Python 在导入模块时会优先在根目录下查找，
# 这样 part_4、part_6 等目录中的脚本 `from part_3.model_modern import GPTModern` 才能找到这个文件。
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# ==========================================
# 完整的现代 GPT 模型（Part 3 及后续 Part 4-9 的核心骨架）
# ==========================================
# 与 Part 2 的 GPT 相比，本模型做了三处关键升级：
#   1. 用 RMSNorm   替换 LayerNorm  （更快、无均值偏移）
#   2. 用 SwiGLU    替换 GELU FFN  （更强的非线性，LLaMA/GPT-4 采用）
#   3. 用 RoPE      替换学习型位置嵌入（更好的外推性，不依赖固定 pos_emb 表）
#   4. 新增 KV Cache（推理时只计算最新 token，历史 K/V 缓存复用，大幅提速）
class GPTModern(nn.Module):
    def __init__(self, vocab_size: int = 256, block_size: int = 256,
                 n_layer: int=4, n_head: int=4, n_embd: int=256, dropout: float=0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True, rope: bool = True,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        # 参数说明：
        #   vocab_size     : 词表大小，字节级 tokenizer 默认 256
        #   block_size     : 模型允许的最大上下文长度（训练时截断用）
        #   n_layer        : Transformer Block 的层数（深度）
        #   n_head         : 注意力头数（Query 头数）
        #   n_embd         : 隐藏层维度（每个 token 用多少个数字表示）
        #   dropout        : 随机丢弃率，训练时防止过拟合，推理时自动关闭
        #   use_rmsnorm    : True → 用 RMSNorm；False → 退回到传统 LayerNorm
        #   use_swiglu     : True → 用 SwiGLU FFN；False → 用普通 GELU FFN
        #   rope           : True → 用 RoPE 旋转位置编码；False → 无位置信息
        #   max_pos        : RoPE 预计算的最大序列长度，影响外推能力（通常远大于 block_size）
        #   sliding_window : 局部注意力窗口大小（None = 全局注意力）。
        #                    设为整数 W 时，每个 token 只关注最近 W 个位置，
        #                    大幅降低长序列的显存消耗（Mistral 等模型的做法）。
        #   attention_sink : "注意力水槽"——强制保留最开头的 K 个 token 在注意力视野中。
        #                    即使开了 sliding_window，这些"锚点" token 也不会被滑出窗口，
        #                    防止模型在极长序列里完全"忘记"开头的重要信息（StreamingLLM 技术）。
        #   n_kv_head      : KV 头数，用于分组查询注意力 (GQA/MQA)。
        #                    None = 与 n_head 相同（标准 MHA）。
        #                    设为比 n_head 小的数（如 n_kv_head=2, n_head=8）时，
        #                    多个 Query 头共享同一对 K/V 头，显存减少、速度更快（LLaMA-3 采用）。
        super().__init__()
        self.block_size = block_size

        # 词嵌入：把 token ID 查表转换为 n_embd 维向量
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # 注意：这里没有 pos_emb！
        # Part 2 的 GPT 需要一张"位置查找表"来告知 token 的顺序，
        # 而现代模型改用 RoPE（旋转位置编码），位置信息直接编码进 Q/K 的旋转角度中，
        # 因此不再需要全局的 pos_emb 嵌入层。
        # self.pos_emb = nn.Embedding(block_size, n_embd)

        self.drop = nn.Dropout(dropout)

        # 堆叠 n_layer 个现代 Transformer Block。
        # 每个 Block 内部集成了：Pre-Norm（RMSNorm 或 LayerNorm）、
        # 带 RoPE 的因果注意力（含 KV Cache 支持）、SwiGLU FFN。
        #
        # 语法：nn.ModuleList([...]) 是 PyTorch 专用的列表容器。
        # 与普通 Python list 不同，它会把里面的子模块正式"登记"到模型中，
        # 使它们的参数能被 .parameters() 遍历到，进而被优化器更新。
        # 如果用普通 list 存放子模块，里面的参数会被优化器完全忽略！
        #
        # `for _ in range(n_layer)` 是列表推导式的循环部分。
        # _ 是 Python 惯例，表示"我不需要这个循环变量"，只是想重复 n_layer 次。
        self.blocks = nn.ModuleList([
            TransformerBlockModern(n_embd, n_head, dropout, use_rmsnorm, use_swiglu, rope, max_pos, sliding_window, attention_sink, n_kv_head)
            for _ in range(n_layer)
        ])

        # 最终归一化层：
        # 如果用 RMSNorm，Block 内部已经在残差之前做了 Pre-Norm（前置归一化），
        # 最后一层输出通常不需要再额外归一化，直接用 Identity（透传，什么都不做）。
        # 如果用 LayerNorm（后置归一化风格），则在最终输出前补一个 LayerNorm。
        #
        # 语法：`A if 条件 else B` 是 Python 三元表达式（内联 if-else），等价于：
        #   if use_rmsnorm:
        #       self.ln_f = nn.Identity()
        #   else:
        #       self.ln_f = nn.LayerNorm(n_embd)
        # nn.Identity() 是一个"透明层"，forward 直接返回输入原值，相当于什么都不做的占位符。
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(n_embd)

        # 输出投影层（Language Model Head）：把 n_embd 维向量投影到 vocab_size 维，
        # 每一维对应词表中一个 token 的"得分"（logit），bias=False 节省参数。
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    # ==========================================
    # 前向传播
    # ==========================================
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list=None, start_pos: int = 0):
        # 参数说明：
        #   idx           : 输入 token 序列，形状 (B, T)
        #   targets       : 目标序列，形状 (B, T)，用于计算 loss；推理时为 None
        #   kv_cache_list : 长度为 n_layer 的列表，每项是对应层的 KVCache 对象（或 None）。
        #                   None 表示不使用缓存，进行完整前向计算（训练 / 第一次推理前馈）。
        #                   非 None 时，每层只计算当前 token 的 Q，并从缓存中读取历史 K/V，
        #                   实现 O(1) 的单步推理（而非重算整个序列）。
        #   start_pos     : 当前输入序列在完整序列中的起始绝对位置。
        #                   训练时始终为 0；使用 KV Cache 时，等于已缓存的 token 数量，
        #                   RoPE 用这个值计算正确的旋转角度（让位置编码与历史序列对齐）。
        # 语法：`B, T = idx.shape` 是元组解包（Tuple Unpacking）。
        # idx.shape 返回形如 (batch_size, seq_len) 的元组，
        # Python 允许把它的两个元素同时赋值给 B 和 T，
        # 比写 B = idx.shape[0]; T = idx.shape[1] 更简洁。
        B, T = idx.shape
        assert T <= self.block_size  # 防止输入超过模型设计的最大上下文长度

        # 只做词嵌入，不加位置嵌入（RoPE 已接管位置编码，不再像 Part 2 那样加 pos_emb）
        x = self.tok_emb(idx)
        # x = x + self.pos_emb(pos)   ← Part 2 的做法，这里已弃用
        x = self.drop(x)

        # 逐层通过 Transformer Block，同时维护每层的 KV Cache。
        #
        # 语法：enumerate(iterable) 同时返回序号和元素，
        # 等价于手写 i=0; for blk in self.blocks: ...; i+=1，但更简洁安全。
        # 这里用 i 来索引 kv_cache_list，取出第 i 层对应的缓存对象。
        new_caches = []
        for i, blk in enumerate(self.blocks):
            # 语法：`A if 条件 else B` 三元表达式。
            # kv_cache_list[i] 用整数索引取列表中第 i 个元素（第 i 层的缓存对象）。
            cache = None if kv_cache_list is None else kv_cache_list[i]
            # 语法：`x, cache = blk(...)` 是元组解包。
            # blk.forward() 返回 (新隐状态, 更新后的KVCache) 这对值，
            # Python 直接把它们分配给左边两个变量，无需写 result = blk(...); x = result[0]。
            x, cache = blk(x, kv_cache=cache, start_pos=start_pos)
            # list.append() 把元素追加到列表末尾，循环结束后 new_caches 长度等于 n_layer。
            new_caches.append(cache)

        x = self.ln_f(x)   # 最终归一化（use_rmsnorm=True 时为 Identity，直接透传）
        logits = self.head(x)  # (B, T, vocab_size)

        # 如果传入了目标 token，就计算交叉熵损失（训练 / 验证时使用）。
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            # 语法：.view(-1, N) 重塑张量形状。
            # -1 告诉 PyTorch"这一维的大小你帮我算"，它会自动用总元素数除以其他维度。
            # logits 形状 (B, T, vocab_size) → view(-1, vocab_size) → (B*T, vocab_size)。
            # cross_entropy 要求输入是 (N, C) 的二维格式，N 是样本数，C 是类别数。
            #
            # 语法：.size(-1) 等价于 .size(最后一维)，这里即 vocab_size。
            # 负数索引与 Python 列表的负索引含义相同：-1 代表倒数第一维。
            #
            # targets.view(-1) 把 (B, T) 展平为 (B*T,)，每个位置对应一个正确的 token ID。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # 与 Part 2 不同，这里返回三个值：logits、loss、以及更新后的 KV Cache 列表。
        # generate 函数在每一步推理后保存 new_caches，下一步直接复用，避免重复计算。
        return logits, loss, new_caches

    # ==========================================
    # 推理生成函数（KV Cache 加速版）
    # ==========================================
    @torch.no_grad()
    def generate(self, 
                 prompt: torch.Tensor, 
                 max_new_tokens=200, 
                 temperature=1.0, 
                 top_k=50, 
                 top_p=None,
                 eos_id=1,  # Part 6 新增：遇到 EOS token 就提前结束生成
                 sliding_window: int | None = None, 
                 attention_sink: int = 0):
        # 参数说明：
        #   prompt         : 提示词 token 序列，形状 (B, T_prompt)
        #   max_new_tokens : 最多生成多少个新 token
        #   temperature    : 采样温度，< 1 输出更保守，> 1 更随机，= 0 变为贪心解码
        #   top_k          : 只保留概率最高的 k 个候选 token（其余设为 -inf 后 softmax 归零）
        #   top_p          : 核采样（Nucleus Sampling）：动态保留累计概率达到 p 的最小候选集
        #   eos_id         : 序列结束符 token ID；所有 batch 都生成了 EOS 则提前退出循环
        #   sliding_window : 局部注意力窗口（此参数已在 Block 内部处理，这里仅作文档保留）
        #   attention_sink : 注意力水槽大小（同上）
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            # 如果 utils 不在路径上，就用一个什么都不做的 lambda 兜底，不报错
            _tk = lambda x, **_: x

        self.eval()  # 关闭 Dropout，进入推理模式
        idx = prompt

        # 初始化 KV Cache 列表：每层对应一个 None（第一步前没有缓存）。
        # 第一次 forward 后，每层会返回自己填充好的 KVCache 对象，取代这里的 None。
        kvs = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # ─── KV Cache 的核心逻辑 ───
            # 第一步（kvs[0] is None）：缓存为空，需要把整个 prompt 喂进去，让模型"读懂"上下文。
            # 后续步骤（kvs[0] 已填充）：历史 K/V 已缓存，只需把最新生成的 1 个 token 喂进去，
            # 计算量从 O(T²) 降至 O(T)，推理速度大幅提升。
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # start_pos 告诉 RoPE 当前这批 token 在完整序列中的起始位置：
            # 第一步为 0（从头开始）；
            # 后续步骤从缓存的 T 属性读出已处理的总 token 数。
            # 语法：`A if 条件 else B` 三元表达式，kvs[0] is None 为 True 时取 0，否则取已处理 token 总数。
            # 注意：这里用 .T 而非 .k.size(2)，因为 RollingKV 启用滑动窗口后缓存会被裁剪，
            #       k.size(2) 会 ≤ sink+window，但 T 始终等于历史 token 总数，正确反映位置偏移。
            start_pos = 0 if kvs[0] is None else kvs[0].T

            # 前向传播，同时传入并更新 KV Cache（kvs 变量在每步被覆盖为最新的缓存）
            # 语法：`logits, _, kvs = self(...)` 是多返回值解包。
            # forward() 返回 (logits, loss, new_caches) 三个值；
            # _ 是 Python 惯例，表示"我知道这里有个返回值但我不需要它"（推理时 loss=None）。
            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            # 语法：logits[:, -1, :] 是三维张量的切片，逗号分隔每一维的索引：
            #   第一维 `:` → 保留所有 batch；
            #   第二维 `-1` → 只取最后一个时间步（序列最后一个位置的预测输出）；
            #   第三维 `:` → 保留所有 vocab 维度。
            # max(temperature, 1e-6) 防止 temperature=0 时发生除以零，1e-6=0.000001 作为兜底下限。
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)  # 温度缩放

            # top-k / top-p 过滤：把不在候选集内的 token logit 置为 -inf
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)

            # 转换为概率分布
            probs = torch.softmax(next_logits, dim=-1)

            # 解码策略：temperature=0 时退化为贪心解码（取概率最大的 token）；
            # 否则按概率分布随机采样，保留生成多样性。
            #
            # 语法：torch.argmax(probs, dim=-1, keepdim=True)
            #   dim=-1 在最后一维（vocab 维）上找最大值的下标；
            #   keepdim=True 保留被压缩的维度，使输出形状为 (B, 1) 而非 (B,)，
            #   与 multinomial 路径的输出形状保持一致，方便后续 torch.cat 对齐。
            #
            # 语法：torch.multinomial(probs, num_samples=1)
            #   按 probs 给出的概率分布随机抽取 1 个样本，概率越高被抽到的可能性越大。
            #   返回的是 token ID（整数索引），形状 (B, 1)，不是概率值本身。
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)

            # 语法：torch.cat([tensor_a, tensor_b], dim=1)
            #   在 dim=1（序列长度维度）上把两个张量首尾相接，
            #   把新生成的 1 个 token 追加到序列末尾。
            #   idx 形状从 (B, T) 变为 (B, T+1)，进入下一轮继续生成。
            idx = torch.cat([idx, next_id], dim=1)

            # Part 6 新增：提前终止。如果所有 batch 的最新 token 都是 EOS，则停止生成。
            # 语法：(next_id == eos_id) 做逐元素比较，返回同形状的布尔张量（True/False）。
            # .all() 检查张量中所有元素是否都为 True（即所有 batch 都已生成 EOS）。
            if eos_id is not None:
                if (next_id == eos_id).all():
                    break

        # 返回完整序列：原始 prompt + 所有新生成的 token，形状 (B, T_prompt + 实际生成数)
        return idx

    # ==========================================
    # 推理生成函数（无缓存版，用于对比调试）
    # ==========================================
    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50, top_p=None,
                sliding_window: int | None = None, attention_sink: int = 0, eos_id: int | None = 1):
        # 本函数与 generate() 功能相同，但故意不使用 KV Cache。
        # 用途：
        #   1. 验证 KV Cache 实现是否正确（两者输出应完全一致）
        #   2. 调试时打印 top-10 候选，方便肉眼检查模型是否在正确预测
        # 代价：每步都要重新计算所有历史 token 的 K/V，时间复杂度 O(T²)，比 generate() 慢得多。
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # 每步都喂入整个（截断后的）窗口，kv_cache_list=None 表示不使用缓存，强制重算所有 K/V。
            idx_cond = idx[:, -self.block_size:]

            # start_pos 设置为窗口起始的绝对位置，让 RoPE 角度与 generate() 的缓存版本对齐，
            # 确保两种实现的位置编码计算结果完全一致，方便数值对比。
            start_pos = idx.size(1) - idx_cond.size(1)

            # 第三个返回值是 new_caches，这里用 _ 忽略（因为不使用缓存）
            logits, _, _ = self(idx_cond, kv_cache_list=None, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)

            # 调试专用：打印概率最高的 10 个 token ID 及其概率值，
            # 对比 generate() 的输出可快速定位 KV Cache 是否引入了误差。
            #
            # 语法：torch.topk(input, k) 返回一个具名元组，包含两个张量：
            #   .values（赋给 topv）：前 k 大的概率值，形状 (B, k)；
            #   .indices（赋给 topi）：对应的 token ID，形状 (B, k)，按概率从高到低排列。
            # Python 支持直接用 `topv, topi = torch.topk(...)` 解包，
            # 无需写 result = torch.topk(...); topv = result.values; topi = result.indices。
            topv, topi = torch.topk(probs, 10)
            print("top ids:", topi.tolist())
            print("top vs:", topv.tolist())

            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # 与 generate() 一致：遇到 EOS 提前终止
            if eos_id is not None and (next_id == eos_id).all():
                break

        return idx


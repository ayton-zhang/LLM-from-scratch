# ==========================================
# 完整的现代 GPT 模型（Part 3 核心骨架，后续 Part 4-9 的基础）
# ==========================================
# 与 Part 2 的 GPT 相比，本模型做了五处关键升级：
#   1. 用 RMSNorm   替换 LayerNorm  （更快、无均值偏移）
#   2. 用 SwiGLU    替换 GELU FFN  （更强的非线性，LLaMA/GPT-4 采用）
#   3. 用 RoPE      替换学习型位置嵌入（更好的外推性，不依赖固定 pos_emb 表）
#   4. 新增 KV Cache（推理时只计算最新 token，历史 K/V 缓存复用，大幅提速）
#   5. 可选分组查询注意力 GQA（多个 Q 头共享 K/V 头，减少显存）
#
# 本文件是 Part 3 的"总装车间"——把所有组件（RMSNorm、RoPE、SwiGLU、
# KV Cache、Block）拼成一个可训练的模型，并提供训练和推理的对外接口。
#
# 文件结构：
#   GPTModern.__init__          → 构建模型架构
#   GPTModern.forward           → 训练 / 一次前向计算
#   GPTModern.generate          → 自回归生成（带 KV Cache 加速）
#   GPTModern.generate_nocache  → 自回归生成（无缓存，用于对比验证）
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
# sys.path.insert(0, ...) 把根目录插入模块搜索路径的最前面，这样 Python 在导入模块时
# 会优先在根目录下查找，part_4、part_6 等目录中的脚本才能 `from part_3.model_modern import GPTModern`。
# Get the absolute path to the folder that contains part_2 and part_3
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

class GPTModern(nn.Module):
    # ==========================================
    # 模型架构：初始化所有组件
    # ==========================================
    def __init__(self, vocab_size: int = 256, block_size: int = 256,
                 n_layer: int=4, n_head: int=4, n_embd: int=256, dropout: float=0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True, rope: bool = True,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        # 参数说明：
        #   vocab_size     : 词表大小，字节级 tokenizer 默认 256（0-255 每个字节一个 token）
        #   block_size     : 模型允许的最大上下文长度（训练时截断用，输入超过此值会报错）
        #   n_layer        : Transformer Block 的层数（深度）。层数越多模型容量越大，
        #                    但也越慢、越容易过拟合
        #   n_head         : 注意力头数（Query 头数），决定从多少个"视角"关注序列
        #   n_embd         : 隐藏层维度（每个 token 用多少个数字表示），信息容量的关键参数
        #   dropout        : 随机丢弃率，训练时随机"打晕"部分神经元防止过拟合，
        #                    推理时（model.eval()）自动关闭
        #   use_rmsnorm    : True → 用 RMSNorm（只除均方根，不减去均值，更快）；
        #                    False → 退回到传统 LayerNorm
        #   use_swiglu     : True → 用 SwiGLU FFN（门控 + Swish 激活，非线性更强）；
        #                    False → 退回到普通 GELU FFN
        #   rope           : True → 用 RoPE 旋转位置编码（通过旋转 Q/K 注入位置信息）；
        #                    False → 无位置信息（退化到"词袋模型"）
        #   max_pos        : RoPE 预计算的最大序列长度，影响外推能力（通常远大于 block_size，
        #                    设为 4096 即使 block_size=256 也没问题）
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

        # 词嵌入层：把 token ID 查表转换为 n_embd 维向量。
        # 直觉类比：tok_emb 是一本"256 页的字典"，每页是一个 n_embd 维的词向量。
        # 输入 token ID=65（ASCII 的 'A'），输出字典第 65 页的向量。
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # 注意：这里没有 pos_emb！
        # Part 2 的 GPT 需要一张"位置查找表"来告知每个 token 在序列中的位置，
        # 而现代模型改用 RoPE（旋转位置编码），位置信息直接编码进 Q/K 的旋转角度中，
        # 因此不再需要全局的 pos_emb 嵌入层。
        # self.pos_emb = nn.Embedding(block_size, n_embd)
        #                ↑ 已弃用，被 RoPE 替代

        # Dropout 层：训练时随机将一部分向量元素置零，
        # 直觉：每次 forward 随机"打晕"一些神经元（dropout=0.1 即 10%），
        # 迫使模型不依赖任何单一的神经元，从而提升泛化能力。
        self.drop = nn.Dropout(dropout)

        # 堆叠 n_layer 个现代 Transformer Block。
        # 每个 Block 内部集成了：Pre-Norm（RMSNorm 或 LayerNorm）、
        # 带 RoPE 的因果注意力（含 KV Cache 支持）、SwiGLU/MLP FFN。
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
        # 如果用 RMSNorm（Pre-Norm 风格），Block 内部已经在残差之前做了归一化，
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
        # 直觉：最后一个 token 的隐状态经过 head 投影，变成"每个词表 token 有多
        #       大概率是下一个 token"的得分分布。
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    # ==========================================
    # forward：训练 / 单次前向计算
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
        #                   RoPE 用这个值计算正确的旋转角度（让新 token 的位置编码
        #                   与历史序列对齐——新 token 是序列第 T_kv 个，不是第 0 个）。
        #
        # 返回值：
        #   logits    : (B, T, vocab_size)，每个位置对词表中每个 token 的预测得分
        #   loss      : 标量，交叉熵损失（targets 为 None 时返回 None）
        #   new_caches: 更新后的 KV Cache 列表（供下一步推理使用）

        # 语法：`B, T = idx.shape` 是元组解包（Tuple Unpacking）。
        # idx.shape 返回形如 (batch_size, seq_len) 的元组，
        # Python 允许把它的两个元素同时赋值给 B 和 T，
        # 比写 B = idx.shape[0]; T = idx.shape[1] 更简洁。
        B, T = idx.shape
        assert T <= self.block_size  # 防止输入超过模型设计的最大上下文长度


        # ─── 第一步：词嵌入 ───
        # 把 token ID 查表转为向量：(B, T) → (B, T, n_embd)。
        x = self.tok_emb(idx)       # (B, T, C)
        # 不再加 pos_emb：RoPE 已在注意力内部替代了位置嵌入的功能。
        # + self.pos_emb(pos)       ← Part 2 的做法，这里已弃用
        x = self.drop(x)            # 训练时随机置零，推理时透传（dropout=0 时无效果）

        # ─── 第二步：逐层通过 Transformer Block ───
        # 数据流：x (B,T,C) → block0 → x' (B,T,C) → block1 → ... → x_final (B,T,C)
        # 每层取出对应的 KV Cache（如果有），计算后把更新后的缓存存回列表。
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

        # ─── 第三步：最终归一化 + 输出投影 ───
        # ln_f：use_rmsnorm=True 时是 Identity（透传），False 时是 LayerNorm。
        x = self.ln_f(x)
        # head：线性投影 (B, T, n_embd) → (B, T, vocab_size)。
        # 输出每个位置对词表中每个 token 的原始得分（logit），未经过 softmax。
        logits = self.head(x)

        # ─── 第四步：计算损失（仅在训练/验证时）───
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            # 语法：.view(-1, N) 重塑张量形状。
            # -1 告诉 PyTorch"这一维的大小你帮我算"，它会用总元素数除以其他维度。
            # logits 形状 (B, T, vocab_size) → view(-1, vocab_size) → (B*T, vocab_size)。
            # cross_entropy 要求输入是 (N, C) 的二维格式，N 是样本数（每个 token 位置
            # 算一个样本），C 是类别数（vocab_size）。
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
    # generate：自回归生成（带 KV Cache 加速）
    # ==========================================
    # 整个生成过程分为两个阶段：
    #
    # Prefill（预填充）阶段：
    #   第一次 forward，把整个 prompt 喂进去，所有层计算完整注意力并缓存 K/V。
    #   这一步相当于"让模型读懂 prompt"。
    #
    # Decode（解码）阶段：
    #   后续每一步只喂入最新的 1 个 token，从缓存读取历史 K/V，
    #   计算量从 O(T²) 降至 O(T)，推理速度大幅提升。
    #   每生成一个 token 就更新缓存，循环直到达到 max_new_tokens 或遇到 EOS。
    #
    # 类比：考试时先读题（prefill），理解题意后每题只需回想关键信息（decode），
    #       不用每道题都重新读一遍整张试卷。
    @torch.no_grad()
    def generate(self,
                 prompt: torch.Tensor,
                 max_new_tokens=200,
                 temperature=1.0,
                 top_k=50,
                 top_p=None,
                 eos_id=1, # addition from part 6 for early stopping
                 sliding_window: int | None = None,
                 attention_sink: int = 0):
        # 参数说明：
        #   prompt         : 提示 token 序列，形状 (1, prompt_len)，batch=1
        #   max_new_tokens : 最多生成多少个新 token（防止无限循环）
        #   temperature    : 温度参数，>1 让分布更"平滑"（输出更随机、更多样），
        #                    <1 让分布更"尖锐"（输出更确定、更保守）。
        #                    =0 表示贪心解码（永远选概率最高的 token）。
        #   top_k          : 只保留概率最高的 k 个 token 参与采样，其余的概率清零。
        #                    减少低概率"噪音"token 的干扰。
        #   top_p          : 核采样（nucleus sampling），按概率从高到低累加，
        #                    只保留累积概率 ≤ p 的那些 token。None 表示不启用。
        #   eos_id         : 结束符 token ID（字节级 tokenizer 中 1 对应特殊结束符）。
        #                    生成到 EOS 时提前停止，不带 EOS 的生成会无意义地续写。
        #   sliding_window : 滑动窗口大小（与 __init__ 中的参数含义相同，
        #                    但这里目前未被使用——保留接口一致性）。
        #   attention_sink : 注意力水槽大小（同上，保留接口一致性）。

        # try/except 处理 top_k_top_p_filtering 的导入：
        # 如果 utils 模块存在就用它做采样过滤，不存在就退化为不做过滤（lambda x, **_: x）。
        # 这样 Part 3 可以独立运行，不依赖后续 Part 的 utils 模块。
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        # model.eval()：切换到推理模式，关闭 Dropout（训练时随机丢弃的神经元在推理时
        # 全部保留），确保同一条 prompt 每次生成的结果完全一致（确定性）。
        self.eval()
        # idx：当前已生成的完整 token 序列（prompt + 已生成 token），形状 (1, T)
        idx = prompt
        # kvs：KV Cache 列表，长度 = n_layer。
        # 初始全为 None（表示还没有任何缓存），prefill 后会全部填充为 KVCache 对象。
        kvs = [None] * len(self.blocks)

        # 语法：`for _ in range(N):` 循环 N 次，_ 是惯例，表示不需要循环变量。
        # 每轮循环生成一个 token，最多 max_new_tokens 轮。
        for _ in range(max_new_tokens):
            # ─── KV Cache 的核心逻辑：决定喂多少 token 给模型 ───
            # Prefill（kvs[0] is None）：缓存为空 → 要把整个 prompt 喂进去建立缓存。
            #   但也要截断到 block_size，防止超长 prompt 溢出，并且只看最新的 block_size 个 token（模型设计的最大上下文）。
            # Decode（kvs[0] 已填充）：历史 K/V 已缓存 → 只需喂最新 1 个 token。
            #   这样计算量恒定为 O(1)，不随序列长度增长。
            # feed full prompt once; then only the last token
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # start_pos：告诉 RoPE 当前 token 在完整序列中的起始绝对位置。
            # Prefill 时为 0（从序列开头开始）。
            # Decode 时用缓存的 K 张量时间维大小（kvs[0].k.size(2)），
            # 即已缓存的 token 总数。这个数告诉 RoPE"新 token 是第 T_kv 个"，
            # 确保旋转角度与历史序列无缝衔接。
            # absolute start position from cache length (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            # 前向传播，返回 (logits, loss, 更新后的 kvs)。
            # 语法：`logits, _, kvs = self(...)` 是多返回值解包，
            # _ 表示"loss 是 None（推理时不传 targets），我不需要它"。
            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            # ─── 采样：从 logits 中选出下一个 token ───
            # 语法：logits[:, -1, :] 是三维张量切片，逗号分隔每一维：
            #   第一维 `:` → 保留所有 batch（此处 batch=1）
            #   第二维 `-1` → 只取最后一个时间步（我们要的"下一个 token 的预测"）
            #   第三维 `:` → 保留所有 vocab 维度
            # /temperature：温度越高，logits 的差异被"抹平"，softmax 后更均匀（更随机）；
            # 温度越低，logits 的差异被"放大"，softmax 后更尖锐（更确定）。
            # max(temperature, 1e-6)：防止除零（temperature=0 时用 1e-6 替代）。
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            # top_k / top_p 过滤：只保留高概率候选 token
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            # softmax 把过滤后的 logits 转成概率分布（所有概率之和=1）
            probs = torch.softmax(next_logits, dim=-1)

            # 贪心解码 vs 随机采样：
            # temperature=0 → 贪心：永远选概率最高的 token（确定性，每次结果相同）
            # temperature>0 → 随机采样：按概率分布随机抽取（非确定性，每次可能不同）
            #
            # 语法：torch.argmax(probs, dim=-1, keepdim=True)
            #   keepdim=True 保留被压缩的维度，使输出形状为 (B, 1) 而非 (B,)，
            #   这样才能与后续 torch.cat([idx, next_id], dim=1) 的形状对齐。
            # 语法：torch.multinomial(probs, num_samples=1) 按概率分布随机抽 1 个 token。
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)

            # 语法：torch.cat([old, new], dim=1) 在序列长度维度拼接，把新 token 追加到末尾。
            idx = torch.cat([idx, next_id], dim=1)

            # ─── EOS 提前终止 ───
            # addition from part 6 for early stopping
            if eos_id is not None:
                # 语法：(tensor == value) 逐元素比较返回布尔张量，.all() 检查是否全为 True。
                # batch=1 时，如果唯一一个样本生成了 EOS，全部停止。
                if (next_id == eos_id).all():
                    break

        # 返回完整序列（prompt + 所有生成的 token，包含 EOS 如果遇到的话）
        return idx

    # ==========================================
    # generate_nocache：自回归生成（无缓存，用于对比验证）
    # ==========================================
    # 功能与 generate() 完全相同，但不使用 KV Cache。
    # 每一步都把"当前完整序列的末尾 block_size 个 token"喂给模型做完整前向计算。
    #
    # 为什么要保留这个"更慢"的方法？
    # - 验证 KV Cache 实现的正确性：两个方法温度 0 时输出应完全一致，
    #   如果不一致说明缓存拼接/裁剪/RoPE start_pos 有 bug。
    # - 代码理解：nocache 版本逻辑更直观（每次完整计算），
    #   适合初学者先理解"到底发生了什么"，再去看 cache 版本的优化。
    # - 调试：出问题时可以用 nocache 版本作 ground truth 逐位置对比。
    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50, top_p=None,
                sliding_window: int | None = None, attention_sink: int = 0):
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # 与 generate() 不同：每次都取完整序列的最后 block_size 个 token，
            # 做完整前向计算（kv_cache_list=None），没有任何缓存复用。
            # always run a full forward over the cropped window, with NO cache
            idx_cond = idx[:, -self.block_size:]

            # start_pos：当前窗口第一个 token 在完整序列中的绝对位置。
            # 如果序列已超出 block_size（如 idx 有 300 token，block_size=256），
            # idx_cond 是最后 256 个 token，start_pos = 300 - 256 = 44，
            # 即第一个 token 在原序列中是第 44 个（从 0 开始算），
            # RoPE 据此正确编码这批 token 的绝对位置。
            # absolute position of first token in the window (matches cached path)
            start_pos = idx.size(1) - idx_cond.size(1)

            # kv_cache_list=None：不使用缓存，每步都完整计算。
            # 语法：`logits, _, _ = self(...)` 三个返回值，后两个用 _ 忽略。
            logits, _, _ = self(idx_cond, kv_cache_list=None, start_pos=start_pos)

            # 采样逻辑与 generate() 完全一致。
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)

            # ─── 调试输出：打印 top-10 候选 token ───
            # 语法：torch.topk(input, k) 返回具名元组，可直接解包为 (values, indices)。
            # 这有助于理解模型在每一步的"思考过程"——它在考虑哪些候选 token、
            # 各自概率是多少。
            topv, topi = torch.topk(probs, 10)
            print("top ids:", topi.tolist())
            print("top vs:", topv.tolist())

            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

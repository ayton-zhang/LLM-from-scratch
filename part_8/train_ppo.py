# ==========================================
# Part 8 核心模块：微型 PPO (Proximal Policy Optimization) 强化学习微调主循环
# ==========================================
# 职责概述：
#   结合预训练/SFT 策略模型、参考模型 (Reference Model) 和 Part 7 奖励模型 (Reward Model)，
#   在单卡上通过在线采样 (On-Policy Rollout) 生成回答，计算标量奖励并应用 KL 惩罚，
#   最后利用 PPO 剪切损失算法对 Policy 网络的 LM 语言模型及 Value Head 价值头进行联合更新。
#
# 整体流程概览（6 个阶段）：
#   A. 在线采样收集：用当前 Policy 对一批 Prompt 生成 Response，Reward Model 打分
#   B. 张量对齐：将不等长序列左截断 + 右 Padding，构造动作掩码
#   C. 对数概率计算：获取旧策略(old)和参考策略(ref)在动作位置上的 logp
#   D. KL 惩罚 + 优势函数：计算 shaped reward = RM 奖励 - KL 系数 × KL 散度
#   E. PPO 更新：用 Clipped Loss + Value Loss 联合更新 Policy + Value Head
#   F. 监控日志：记录 KL 偏移量和 loss 指标
#
# 与标准 RLHF 的区别（本教程的简化点）：
#   1. 优势估计使用”即时奖励 - Value”的简化近似，而非完整的 GAE 时序差分展开
#      （完整 GAE 需要从序列末尾反向递推，代码量更大但对教学不友好）
#   2. 奖励为稀疏奖励（仅序列末尾有标量值），而非每个 token 都有稠密奖励
#   3. 批次大小很小（默认 4），适合单卡调试和教学演示
# ==========================================

from __future__ import annotations
import argparse, torch
from pathlib import Path

# import torch
# torch.manual_seed(0)  # 保持注释，如需复现可取消注释以固定随机种子

# ─── 本模块内部导入 ───
# PolicyWithValue: 在 SFT 语言模型基础上加上 Value Head（价值头），
#   使得同一个 backbone 能同时输出 token 概率（给 actor）和状态价值（给 critic）。
#   这比”两个独立网络分别做 actor 和 critic”更省显存、参数共享更充分。
from policy import PolicyWithValue

# RLHFTokenizer: 继承 Part 6 的 Tokenizer，额外提供 encode/decode 和特殊 token 处理
# format_prompt_only: 将原始 prompt 包装为模型输入格式（带 <s> 等特殊标记）
# format_example: 将 (prompt, response) 拼接为完整的训练样本文本
# sample_prompts: 从内置的小型 prompt 池中随机采样一批问题文本
# gather_logprobs: 从 log_softmax 输出中按标签索引提取条件对数概率
# shift_labels: 将 labels 左移一位，使位置 t 的标签对应位置 t 的预测
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels

# model_logprobs: 对给定模型和输入序列，计算每个位置预测下一个 token 的条件对数概率
#   返回形状 (B, T-1)，第 t 个元素 = log P(token_{t+1} | token_1, ..., token_t)
from rollout import model_logprobs

# ─── 跨模块导入 Part 7 的奖励模型 (Reward Model) ───
# 设计决策：Part 8 的 PPO 训练依赖 Part 7 已训练好的 Reward Model 来给生成结果打分。
# 但 Part 7 和 Part 8 是兄弟目录，不在彼此的 Python 搜索路径中。
# 因此通过 sys.path.append 动态将 part_7 目录加入模块搜索路径。
#
# 语法：sys.path.append(...) 动态将父目录中的 part_7 路径添加到 Python 模块搜索路径列表中，
# 使得当前脚本能够成功 import part_7/model_reward.py 中定义的 RewardModel 类。
# 语法：Path(__file__).resolve().parents[1] 获取当前文件的上上级路径（即项目根目录），拼接 'part_7'。
#   Path.resolve() 将相对路径转为绝对路径；.parents[1] 取第 1 级父目录（从 0 开始编号：0=当前文件所在目录 part_8，1=项目根目录）。
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402  # 语法：noqa: E402 告知 flake8 忽略”import 未放在文件最顶部”的 PEP8 警告

# ppo_losses: 计算 PPO 的三项损失——Policy Loss (clipped)、Value Loss (MSE)、
#   以及可选的 Entropy Bonus（熵奖励，用于鼓励探索），最终汇总为 Total Loss。
from ppo_loss import ppo_losses


# ==========================================
# 辅助函数：针对 (Prompt, Response) 文本计算奖励模型标量得分
# ==========================================
# 作用：将一个 (prompt, response) 对送入冻结的 Reward Model，得到单一的标量奖励值。
# 这个奖励值将作为 PPO 优化的"环境信号"——Policy 生成的高质量回答会得到高分，
# 低质量/有害回答会得到低分，从而引导 Policy 向人类偏好的方向更新。
#
# 类比理解：
#   Reward Model 就像是"AI 评委"，读完 prompt+response 后打一个分数。
#   PPO 训练的目标就是让 Policy（选手）学会写出评委喜欢的高分答案。
#
# 参数说明：
#   reward_model: Part 7 训练好的 RewardModel 实例，参数冻结，仅用于推理打分
#   tok:          RLHF 专用分词器，负责将文本转为 token ID 序列
#   prompt:       原始问题文本（如 "解释什么是梯度下降"）
#   response:     Policy 模型生成的回答文本
#   device:       硬件设备（'cuda' 或 'cpu'）
# 返回值：
#   float: 标量奖励分数，越高表示回答质量越好
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    # 1. 格式化样本：将 prompt 与 response 拼接为符合 Part 6/7 规范的 Example 格式文本
    #    这一步确保 Reward Model 看到的输入格式与它训练时完全一致（同样的特殊 token、同样的拼接方式）。
    # 语法：__import__('part_6.formatters', fromlist=['Example']) 动态导入 part_6 的 Example 类。
    #   __import__(name, fromlist=[...]) 是 Python 内置的 import 机制——当 fromlist 非空时，
    #   返回最右边的子模块（part_6.formatters），等价于 import part_6.formatters 后的模块对象。
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))

    # 2. Token 编码与截断：转化为 token ID 序列，并截断至模型支持的最大 block_size 长度
    #    tok.encode() 内部会自动添加 <s>（句首）和 </s>（句尾）特殊标记。
    ids = tok.encode(text)

    # 3. 构造输入张量：加上 batch 维度，形状从 (seq_len,) 变为 (1, seq_len)
    #    ids[:tok.block_size] 做右截断——如果序列过长，只保留前 block_size 个 token
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)

    # 4. 前向传播：在 no_grad 模式下调用奖励模型，获得标量 reward 输出，并转换为 Python float 浮点数
    # 语法：with torch.no_grad(): 禁用梯度追踪——Reward Model 参数冻结且只做推理，
    #   不保留计算图，大幅减少显存开销并加快推理速度。这是 PyTorch 推理的标准写法。
    with torch.no_grad():
        r = reward_model(x)  # 输出形状 (1,) 的张量，包含一个标量奖励值
    # r[0] 取 batch 中第 0 个样本的奖励张量，.item() 将 0 维张量转为 Python float
    return float(r[0].item())


# ==========================================
# PPO 训练主函数 (Main Training Loop)
# ==========================================
# 这是整个 Part 8 的入口——它串联了 RLHF 的完整流程：
#   加载模型 → 采样生成 → 奖励打分 → 计算优势 → PPO 更新 → 保存结果
#
# 核心概念速览（RL 术语 → NLP 对应）：
#   State（状态）      = Prompt 文本 + 已生成的部分 Response
#   Action（动作）     = 生成下一个 token
#   Policy（策略）     = 语言模型的 token 概率分布
#   Reward（奖励）     = Reward Model 对完整回答的打分（稀疏，仅末尾有值）
#   Value（状态价值）  = Value Head 预测的”从当前状态出发，未来能获得多少总奖励”
#   Advantage（优势）  = 实际奖励 - 预测价值，正值表示”这一步比预期好”
def main():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    # 针对 PPO 训练流程中的关键超参数进行解析设置。
    # 每个参数的含义和设计考量见下方逐条注释。
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo', help='模型检查点保存目录')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6) -- Policy 和 Reference 模型的初始化来源')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7) -- 用于给生成结果打分')
    p.add_argument('--steps', type=int, default=100, help='总训练步数 -- 每步采样一批 prompt、生成回答、做一次 PPO 更新')
    p.add_argument('--batch_size', type=int, default=4, help='每步采样的 Prompt 批次大小 -- 小 batch 适合教学，大 batch 训练更稳定')
    p.add_argument('--block_size', type=int, default=256, help='Transformer 序列最大长度上限 (含 prompt + response)')
    p.add_argument('--resp_len', type=int, default=64, help='Policy 自动生成 Response 的最大 token 数')
    # kl_coef 是 RLHF 中最关键的平衡参数之一：
    #   设太小 -> Policy 可能通过生成乱码 (Reward Hacking) 骗取高分
    #   设太大 -> Policy 被 Reference 束缚太紧，学不到新行为
    p.add_argument('--kl_coef', type=float, default=0.01, help='KL 散度惩罚系数，防止 Policy 偏离 Ref 策略过远')
    p.add_argument('--gamma', type=float, default=1.0, help='强化学习奖励折扣因子 -- 1.0 表示未来奖励与即时奖励同等重要')
    p.add_argument('--lam', type=float, default=0.95, help='GAE (Generalized Advantage Estimation) lambda 参数 -- 控制偏差-方差权衡')
    p.add_argument('--lr', type=float, default=1e-5, help='AdamW 优化器学习率 -- RL 训练通常用比 SFT 更小的学习率以保证稳定')
    p.add_argument('--bpe_dir', type=str, default=None, help='BPE 分词器目录路径')
    p.add_argument('--cpu', action='store_true', help='是否强制使用 CPU 进行训练')
    args = p.parse_args()

    # 语法：`A if 条件 else B` 是 Python 三元表达式（内联 if-else），根据条件返回两个值之一。
    # 此处用于自动判断可用设备：有 GPU 且未强制 CPU → 用 cuda；否则用 cpu。
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer：初始化 RLHF 专用分词器，负责文本 ↔ token ID 的双向转换
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # ==========================================
    # 2. 加载 SFT 检查点并初始化 Policy 与 Reference 模型
    # ==========================================
    # 设计决策：为什么需要两个完全相同的模型（Policy 和 Reference）？
    #   - Policy（可训练）：在 SFT 权重基础上持续更新，学习如何生成高分回答
    #   - Reference（冻结）：始终保持在初始 SFT 状态，作为”不敢偏离太远”的基准锚点
    #   这就像学生（Policy）可以自由发挥，但老师（Reference）会时不时提醒”别跑太偏”。
    #
    # 语法：torch.load(..., map_location=device) 将保存的模型权重直接加载至指定的硬件设备（CPU/GPU）上，
    #   避免先加载到 CPU 再 .to(device) 的多余拷贝。
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    # cfg.get(key, default) 安全地读取配置字典，若键不存在则使用默认值
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    # 2.1 初始化可优化的 Policy 模型（包含语言模型 LM 与价值头 Value Head）
    # PolicyWithValue 的结构：在普通 GPT LM 的基础上加了一个线性层（Value Head），
    #   将每个位置的隐状态映射为一个标量”状态价值”。
    #   为什么 Value Head 和 LM 共享 backbone？——共享表示学习，节省参数和显存，
    #   且 Value 能从 LM 已经学会的语言理解能力中受益。
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # 使用 Part 6 SFT 权重初始化 LM 部分

    # 2.2 初始化冻结的 Reference 模型（基准参考模型）
    # 设计决策：RLHF 过程中模型极易通过生成语法异常但高分的文本来”欺骗”奖励模型（Reward Hacking）。
    # 例如 Policy 可能学会反复输出”太棒了！非常好！”这种空洞但 Reward Model 喜欢的文本。
    # KL 惩罚通过限制 Policy 输出分布不偏离原始 SFT 太远，有效抑制这种作弊行为。
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    # 逐个参数设置 requires_grad=False，确保 ref 不参与任何梯度计算
    for p_ in ref.parameters():
        p_.requires_grad_(False)  # 冻结参数，不参与反向传播计算梯度
    ref.eval()  # 设为评估模式，关闭 Dropout 等训练专用行为

    # ==========================================
    # 3. 加载 Part 7 训练好的奖励模型 (Reward Model)
    # ==========================================
    # Reward Model 是整个 RLHF 流程的”裁判”——它不参与训练，只负责给 Policy 的生成结果打分。
    # 它的评分信号是 Policy 学习的唯一驱动力，因此 Reward Model 的质量直接决定了 PPO 微调的上限。
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()  # 奖励模型仅作为固定评估器，冻结参数并开启 eval 模式

    # ==========================================
    # 4. 初始化 AdamW 优化器
    # ==========================================
    # 只优化 policy.parameters()——Reference 和 Reward Model 都不参与梯度更新。
    # betas=(0.9, 0.999) 是 Adam 的标准默认动量参数，控制一阶和二阶矩估计的指数衰减率。
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ==========================================
    # 5. 构建微型 Prompt 提示池
    # ==========================================
    # 从内置题库中随机抽取 16 条 prompt 作为训练素材。
    # 在真实 RLHF 中，prompt 池通常包含数万条多样化的指令，
    # 这里是教学简化版——用少量固定 prompt 演示完整流程。
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        # ==========================================
        # 阶段 A：在线采样批次收集 (On-Policy Rollout Generation)
        # ==========================================
        # 这是 PPO "在线策略"(On-Policy) 的核心体现：
        #   每次更新前，必须用当前最新的 Policy 重新采样生成数据。
        #   不能复用旧的生成数据——因为 Policy 参数已经变了，旧数据不再代表当前策略的行为。
        #   这一点与监督学习（可以反复用同一批数据）有本质区别。
        #
        # 类比理解：
        #   就像棋手（Policy）每下一局（rollout）后复盘改进，下一局时棋力已经变了，
        #   所以必须用新棋力重新下棋来获取新的对局记录，而不能用旧棋力下的旧棋谱。
        #
        # 1. 循环切片选择当前 Batch 的 Prompt 文本
        #    语法：列表切片 [start:end]，当 end < start 时返回空列表（跨过列表尾部时触发）。
        #    下面的 if 分支处理这种情况：从列表头部补齐不足的部分。
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts) ]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        # format_prompt_only 将原始文本包装为模型所需的格式（添加 <s> 等），
        # .replace("</s>", "") 去掉格式模板自带的句尾标记，因为 response 还没生成。
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]  # 每条 prompt 转为 token ID 列表

        # 2. 调用当前 Policy 模型自回归生成 Response 序列
        #    为什么用 torch.no_grad()？——采样生成阶段不需要记录梯度（梯度在后面的 PPO 更新阶段才需要），
        #    此处只收集"环境数据"（就像强化学习中的 agent 与环境交互），关闭梯度节省显存。
        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)  # 形状: (1, prompt_len)
                # temperature=0.2：低温度使分布更"尖锐"（接近 greedy），生成结果更确定、质量更高
                # top_k=3：每步只从概率最高的 3 个候选 token 中采样，过滤掉低概率的噪声 token
                # 这两个参数共同控制"探索 vs 利用"的平衡——温度越低、top_k 越小，越接近确定性解码
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)
                # out 形状 (1, total_len)，.tolist() 转为 Python 列表方便后续处理
                out_ids.append(out[0].tolist())

        # 3. 划分 Prompt 与 Response 边界，并利用 Reward Model 计算环境标量奖励
        #    对每条样本：(完整序列, prompt长度边界, RM奖励分数)
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]  # 完整序列 = prompt tokens + response tokens
            # 左截断：如果 prompt 超过 block_size 限制，只保留最后 block_size 个 token
            #   p_ids 是实际馈入模型的 prompt 部分，boundary 就是 prompt 结束的位置索引
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            # 切片：从 boundary 位置到末尾的 token 就是模型生成的 response 部分
            resp_ids = full[boundary:]

            # 将生成的 response 解码为文本，并送入 Reward Model 评分
            # 注意：奖励是稀疏的——只在整个 response 生成完毕后给一个总分数，
            # 而不是每个 token 都有一个即时奖励。这是 RLHF 的常见做法。
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        # ==========================================
        # 阶段 B：张量对齐、左截断 Padding 填充与动作掩码构造
        # ==========================================
        # 核心问题：批次内每条序列长度不相等（不同 prompt 长度 + 不同 response 长度），
        # 但 PyTorch 的批处理要求所有序列对齐到相同长度。这里采用"左截断 + 右 Padding"策略。
        #
        # 为什么是左截断（丢弃开头 token）而非右截断（丢弃末尾 token）？
        #   因为序列末尾是新生成的 response token——这是 PPO 需要学习的关键部分，
        #   绝不能丢弃！而序列开头是 prompt 的老内容，丢弃部分影响较小。
        #
        # 确定批次内最大序列长度 max_len（受限于 Policy 的上下文窗口 block_size）
        # 语法：getattr(obj, "attr_name", default) 安全获取属性，若不存在则返回默认值。
        policy_ctx = getattr(policy, "block_size", block_size)
        # max(t[0].numel() for t in data)：获取批次中最长序列的 token 数
        # .numel() 返回张量中元素总数（number of elements），在此即序列长度
        max_len = min(policy_ctx, max(t[0].numel() for t in data))
        B = len(data)  # 批次大小

        # 预分配四个张量缓存，一次性分配整块 GPU 显存比循环中逐个 .cat() 拼凑更高效：
        seq     = torch.zeros(B, max_len, dtype=torch.long, device=device)       # Token 序列张量, 形状 (B, max_len)
        mask    = torch.zeros(B, max_len, dtype=torch.bool, device=device)      # Response 动作掩码, 形状 (B, max_len)
        # 补充：mask[b, t] = True 表示位置 t 的 token 是"模型自己生成的 response 部分"，
        #   只有这些位置的 token 才参与 PPO 损失计算；prompt 部分不参与（不是策略的动作）。
        last_idx = torch.zeros(B, dtype=torch.long, device=device)              # 每条序列末尾索引, 形状 (B,)
        rewards  = torch.zeros(B, max_len, dtype=torch.float, device=device)    # 稀疏奖励张量（仅序列结尾有值）, 形状 (B, max_len)

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()                    # 原始序列的真实长度（可能超过 max_len）
            L = min(L_full, max_len)                # 截断后的有效长度
            drop = L_full - L                       # 从左侧被裁切丢弃的 token 数量
            b = max(0, boundary - drop)             # 左裁切后重新计算的 prompt/response 边界索引
            # 关键：ids[-L:] 取"最后 L 个 token"实现左截断——丢弃开头的 (L_full - L) 个 token
            seq[i, :L] = ids[-L:]                   # 填入截断后的序列
            if L < max_len:
                # 语法：右 Padding——将短于 max_len 的尾部位置填充为 <pad> token，使该位置在
                #   注意力计算中被忽略（通常配合 attention_mask 使用，此处用 ID=2 占位）
                seq[i, L:] = 2  # <pad> token ID = 2
            # 关键：mask[i, b:L] 只标记 Response 部分的 token 为 True
            #   b 是 prompt/response 分界线，L 是序列有效长度
            #   这样在后续计算 loss 时只会对模型"自己的动作"（生成的 token）求梯度
            mask[i, b:L] = True
            # 稀疏奖励：标量奖励 r_scalar 放在回答序列的最后一个有效 token 位置上
            rewards[i, L-1] = r_scalar
            last_idx[i] = L-1

        # ==========================================
        # 阶段 C：计算旧策略 (Old Policy) 与参考策略 (Ref) 的对数概率与状态价值
        # ==========================================
        # 为什么需要"旧"对数概率 old_logp？
        #   PPO 的核心思想是用"重要性采样比率" ratio = π_new / π_old 来衡量策略更新幅度。
        #   如果 ratio 偏离 1 太远（即新策略与旧策略差异过大），就用 clip 限制更新步长。
        #   因此必须保存采样时刻（更新前）的 logp 作为"旧的基准"，在更新后再算一次"新的"做对比。
        #
        # model_logprobs 返回形状 (B, T-1) 的张量——注意少了 1 个时间步！
        #   原因：语言模型在位置 t 的输出预测的是位置 t+1 的 token，
        #   所以 logprobs[t] = log P(token_{t+1} | token_{1...t})，共 T-1 个预测。
        # 语法：model_logprobs 内部调用 model(seq) 得到 logits，再做 log_softmax + gather 提取对数概率。
        pol_lp = model_logprobs(policy, seq)  # 形状: (B, T-1)，旧策略的对数概率
        ref_lp = model_logprobs(ref, seq)     # 形状: (B, T-1)，参考策略的对数概率

        # 前向传播计算序列中各个位置的状态价值 (Value Estimates)
        # Value Head 对每个位置输出一个标量，预测"从这个状态开始，未来能获得的总奖励"。
        # 这类似于围棋中评估当前盘面的"胜率"——Value 越高，说明当前位置越有利。
        # 语法：logits, values, _ = policy(seq, None) 接收 (B, T) 输入，返回三个值的元组。
        #   _ 是 Python 惯例——占位符，表示"我知道这里有第三个返回值（KVCache），但我不需要它"。
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        # values 原始形状 (B, T)，[:, :-1] 丢弃最后一个位置，使 values 形状 (B, T-1) 与 pol_lp 对齐
        # 为什么丢弃？因为 position T 的 value 预测超出序列范围，没有对应的 token 需要评估。
        values = values[:, :-1]

        # 仅筛选动作 (Action Tokens, 即 Response 部分) 的对数概率与价值估计
        # 关键理解：PPO 只优化"策略做出的动作"——也就是模型自己生成的 response token。
        #   Prompt 部分的 token 是环境给定的，不是策略的选择，不该参与 actor 的损失计算。
        #
        # 语法：act_mask = mask[:, 1:] —— 为什么从第 1 列开始切片？
        #   mask 的形状是 (B, T)，标记了每个"实际 token"是否是 response 部分。
        #   但 logprobs 的形状是 (B, T-1)，其第 t 个元素对应"位置 t 预测位置 t+1"。
        #   为了对齐，mask 的第 1~T-1 列（即 mask[:, 1:]）正好对应"预测位置 1~(T-1) 的 token"。
        #   也就是说，mask[0] 对应 token_0 但 logprobs 第 0 个预测的是 token_1，所以要用 mask[1:]。
        #   更直观的理解：token_0（通常是 <s>）从来不被任何位置预测，所以 mask 的第 0 列需要排除。
        act_mask = mask[:,1:]  # 形状: (B, T-1)，True 的位置 = "该预测对应的是 response token"
        # 布尔索引筛选：act_mask 为 True 的位置被拉平为一维向量
        # .detach() 切断梯度——旧策略的概率在 PPO 更新中作为常数基准，不应回传梯度
        old_logp   = pol_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        ref_logp   = ref_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        old_values = values[act_mask].detach()    # 形状: (N_action_tokens,)

        # ==========================================
        # 阶段 D：KL 惩罚、塑造奖励 (Shaped Rewards) 与优势函数 (Advantage) 计算
        # ==========================================
        # 这是 RLHF 的"奖励塑造"(Reward Shaping) 核心步骤，分三步走：
        #
        # 1. 计算每个动作 token 的 KL 散度近似值
        #    KL 散度衡量两个概率分布之间的"距离"——分布越接近，KL 越接近 0。
        #    公式：KL(π_old || π_ref) ≈ E_{a~π_old}[log π_old(a|s) - log π_ref(a|s)]
        #    这里用单样本近似：KL ≈ log π_old(a|s) - log π_ref(a|s)，其中 a 是实际采样的 token。
        #    每个 token 是独立的离散动作，所以直接逐元素相减即可。
        #
        #    注意：此处的 KL 是基于"每个采样的 token 动作"计算的近似值。
        #    严格来说 KL 需要对整个词表求和，但那只在"需要完整的分布距离"时才必要，
        #    在 PPO 中，用采样的 token 估计 KL 计算量小、效果足够好（InstructGPT 的做法）。
        kl = (old_logp - ref_logp)  # 形状: (N_action_tokens,)

        # 2. 塑造奖励 (Shaped Reward)：在原始 RM 标量奖励的基础上，减去 KL 惩罚项
        #    核心思想：Reward Model 只看最终回答好坏，不关心语言是否自然流畅。
        #    如果不加 KL 惩罚，Policy 会"走火入魔"——生成一堆语法错乱但 RM 喜欢的高分词。
        #    加了 KL 惩罚后，每偏离一次 reference，就扣一次分，迫使模型保持自然的语言风格。
        #
        #    语法：rewards[:, 1:] 取 rewards 张量的第 1 列到最后一列（形状 B, T-1），
        #    与 act_mask 对齐（logprobs 对应的是预测时刻，形状为 B, T-1）。
        #    [act_mask] 布尔索引提取出 response 位置的奖励值。
        shaped_r = rewards[:,1:][act_mask] - args.kl_coef * kl  # 形状: (N_action_tokens,)

        # 3. 优势函数 (Advantage) 与目标回报 (Returns) 估计
        #    优势函数 A(s,a) 回答的问题是："在当前状态 s 下做了动作 a，比预期的好多少？"
        #    A > 0 → 这个动作比预期好，增大它的概率
        #    A < 0 → 这个动作比预期差，减小它的概率
        #    A ≈ 0 → 中规中矩，不调整
        #
        #    教程简化版：将 shaped_r 直接作为 returns（目标回报），adv = returns - values。
        #    完整的 GAE 算法需要从序列末尾反向递推，累积未来奖励的加权和（用 gamma 和 lambda 参数）。
        #    这里的简化等价于：假设 gamma=0（只看即时奖励，不考虑未来）。
        #    对于 token 级别的生成任务，即时 KL 惩罚已经提供了逐 token 的稠密信号，
        #    所以即使不做完整 GAE 递推，训练也能正常进行。
        returns = shaped_r                     # 形状: (N_action_tokens,)
        adv = returns - old_values             # 形状: (N_action_tokens,)，优势 = 实际奖励 - 预测价值

        # 优势归一化：对整批动作的优势做 Z-Score 归一化（零均值 + 单位方差）
        # 为什么需要归一化？PPO 的 clip 范围（默认 ±0.2）是固定值。
        #   如果原始优势的数值范围变化很大（有时 ±0.01，有时 ±100），
        #   clip 的效果就无法保持一致——有时太松、有时太紧。
        #   归一化后优势永远在相近的数量级，clip 参数可以稳定发挥作用。
        #
        # 语法：(adv - adv.mean()) / (adv.std().clamp_min(1e-6))
        #   .mean() 和 .std() 对一维张量求均值和标准差，返回标量。
        #   .clamp_min(1e-6) 将 std 下限钳制为 1e-6，防止所有优势完全相同（std=0）时除零报错。
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

        # ==========================================
        # 阶段 E：PPO 损失计算与梯度更新 (PPO Update Pass)
        # ==========================================
        # 这是 PPO 的"演员-评论家"(Actor-Critic) 联合更新步骤：
        #   Actor（LM head）：学习生成更受 Reward Model 青睐的 token
        #   Critic（Value head）：学习更准确地预测未来的总奖励
        #   两者共享同一个 Transformer backbone，梯度同时流向共享层。
        policy.train()  # 切换至训练模式——启用 Dropout 等训练行为

        # 前向传播：用更新前的参数重新计算当前策略的对数概率与价值
        # 注意：虽然是"new"logp，但此时参数还没更新（optimizer.step() 在后面），
        #   所以"new_logp"其实还是当前的、与 old_logp 相同的分布（第一次迭代时）。
        #   真正产生差异是在多 epoch 的 PPO update 中——第二次 forward 时参数已变，
        #   new_logp 就不等于 old_logp 了。本教程单步更新，差异不大。
        #
        # 语法：policy(seq, None) 的三返回值解包：
        #   logits:  (B, T, vocab_size) — 词表上的原始得分（未归一化）
        #   values:  (B, T) — 每个位置的标量状态价值
        #   _:       KVCache — 此处不需要，用 _ 占位丢弃
        logits_new, values_new_full, _ = policy(seq, None)  # 形状: logits (B, T, V), values (B, T)

        # torch.log_softmax(logits, dim=-1)：将 raw logits 转为对数概率
        #   相比先 softmax 再 log，log_softmax 在数值上更稳定（避免了 softmax 中的 exp 溢出）。
        #   logits_new[:, :-1, :] 丢弃最后一个位置的预测（position T），形状变为 (B, T-1, V)。
        #   dim=-1 表示在词表维度做归一化（每行所有词表 token 的概率 log 之和=0）。
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)  # 形状: (B, T-1, vocab_size)
        labels = seq[:, 1:]  # 形状: (B, T-1)，每个位置的真实下一个 token ID（作为预测目标）

        # 从 vocab_size 个对数概率中提取"真实标签 token"对应的那个值
        # 语法：.gather(dim, index) 沿 dim 维度按 index 索引收集元素。
        #   logp_full 形状 (B, T-1, V)，labels 形状 (B, T-1)
        #   → labels.unsqueeze(-1) 将 labels 变为 (B, T-1, 1)，作为 gather 的索引
        #   → gather 沿 dim=-1（词表维）查找：对每个 (b, t) 位置，取 labels[b,t] 对应的 logp
        #   → .squeeze(-1) 去除最后的多余维度，恢复为 (B, T-1)
        #   最终结果：new_logp_all[b, t] = log P(token_{t+1} | token_{1..t})，即条件对数概率
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # 形状: (B, T-1)

        # 提取 Action 掩码位置的最新对数概率与价值估计
        # 与阶段 C 中 old_logp 的提取方式完全一致，确保对比的是同一批 token
        new_logp   = new_logp_all[act_mask]          # 形状: (N_action_tokens,)
        new_values = values_new_full[:, :-1][act_mask]  # 形状: (N_action_tokens,)

        # 调用 ppo_losses 核心算法计算三项损失
        # 参数说明：
        #   new_logp, old_logp: 新旧策略的对数概率，用于计算重要性比率 ratio = exp(new - old)
        #   adv: 优势函数，正值→增大概率，负值→减小概率
        #   new_values, old_values: 新旧价值估计（old_values 用于计算 value clipping）
        #   returns: 目标回报（shaped_r），Value Head 的回归目标
        #   clip_ratio=0.2: PPO 的核心参数——限制策略更新幅度不超过 ±20%
        #     如果 ratio ∈ [0.8, 1.2]，直接优化；超出范围则用 clip 截断梯度
        #   vf_coef=0.5: Value Function 损失在总损失中的权重系数
        #   ent_coef=0.0: 熵奖励系数（此处关闭，如需鼓励更多探索可设为正值）
        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        # out_loss 是具名元组，包含 .policy_loss, .value_loss, .entropy_loss, .total_loss
        loss = out_loss.total_loss

        # 标准 PyTorch 训练三步曲：清零梯度 → 反向传播 → 参数更新
        # 语法：opt.zero_grad(set_to_none=True) 将梯度缓冲设为 None。
        #   set_to_none=True 比默认的 zero_grad()（置为全零张量）更省显存：
        #   None 不占内存而零张量占；PyTorch 在 backward() 时对 None 梯度会自动分配，
        #   对于稀疏更新的参数（如嵌入层只更新被用到的行），这能节省可观的显存和计算。
        opt.zero_grad(set_to_none=True)
        loss.backward()  # 反向传播计算梯度——梯度从 total_loss 出发，沿计算图回溯至 policy 各参数

        # 梯度裁剪：将所有权重的梯度总范数钳制在 1.0 以内
        # 为什么 RL 训练特别需要梯度裁剪？
        #   RL 的信号比监督学习噪声大得多（奖励是稀疏的、由另一个模型给的），
        #   优势函数的方差也更大，容易出现单个 batch 产生巨大梯度→参数崩坏的情况。
        #   梯度裁剪是 RL 训练的"安全阀"。
        # 语法：torch.nn.utils.clip_grad_norm_(parameters, max_norm)，末尾 _ 表示原地操作。
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()  # AdamW 根据裁剪后的梯度更新策略模型的所有可训练参数
        policy.eval()  # 切回评估模式——关闭 Dropout，为下一轮采样做准备

        # ==========================================
        # 阶段 F：KL 偏移监控与指标日志记录
        # ==========================================
        # RLHF 训练的难点之一：训练过程不可见——loss 下降不代表模型在变好。
        #   因为 RM 奖励是模型自己生成的，loss 降低可能只是 Value Head 学会了"预测自己的预测"。
        #   因此需要额外的监控指标来判断训练是否健康。
        #   最关键的两个指标就是下面监控的 KL 散度。
        with torch.no_grad():
            # 1. KL(old || new)：监控一次 PPO 更新后，策略相比采样时的旧策略偏离了多少
            #    数学含义：E_{a~old}[log π_old(a) - log π_new(a)]，衡量单步更新的幅度
            #    健康范围：通常 KL_move 应该在 0.001 ~ 0.05 之间
            #    过大（> 0.1）→ 学习率太大或梯度爆炸，策略变化过于剧烈
            #    过小（< 1e-5）→ 学习率太小，几乎没学到任何东西
            lp_post = model_logprobs(policy, seq)          # 更新后用新参数重新算一次 logp，形状: (B, T-1)
            lp_post = lp_post[act_mask]                    # 仅选动作位置，形状: (N_action_tokens,)
            # 近似 KL(old||new) = E[log π_old - log π_new]
            # 注意：old_logp 是阶段 C 中 .detach() 后的常数，不会随参数变化
            kl_post = (old_logp - lp_post).mean()          # 标量，衡量"这一步更新跑了多远"

            # 2. KL(now || ref)：监控当前更新后的策略与冻结基准 reference 模型之间的总偏离距离
            #    数学含义：E_{a~now}[log π_now(a) - log π_ref(a)]
            #    健康范围：KL_ref 应缓慢增长，训练结束时通常在 0.01~0.5 之间
            #    过大（> 1.0）→ Policy 已严重偏离原始语言模型，可能退化成语无伦次
            #    如果 KL_ref 持续不变而 loss 在降 → 可能 reward hacking 失败，模型没学到新行为
            lp_now = lp_post                                # 就是上面刚算的更新后 logp
            kl_ref_now = (lp_now - ref_logp).mean()         # 标量，衡量"相比初始 SFT 偏了多远"

        step += 1
        if step % 10 == 0:
            # 每 10 步打印一次关键指标：
            #   loss:         总损失（policy loss + value loss 的加权和）
            #   value loss:   Value Head 的 MSE 损失——衡量"预测价值 vs 实际回报"的差距
            #   KL_move:      单步策略偏移量——判断学习率是否合适
            #   KL_ref:       累计与初始模型的偏差——判断训练是否在健康轨道上
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| value loss {out_loss.value_loss.item():.4f} | KL_move {kl_post.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )

    # ==========================================
    # 6. 保存 PPO 微调后的最终模型检查点
    # ==========================================
    # 保存格式与 Part 6 SFT 检查点一致：字典包含 'model' 和 'config' 两个键。
    # policy.state_dict() 包含 LM backbone 和 Value Head 的所有参数。
    # 语法：Path.mkdir(parents=True, exist_ok=True) 递归创建目录，exist_ok=True 避免目录已存在时报错。
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved PPO policy to {args.out}/model_last.pt")

# ==========================================
# 脚本入口：Python 标准守卫模式
# ==========================================
# 语法：if __name__ == '__main__': 是 Python 的惯用模式。
#   当文件被 `python train_ppo.py` 直接运行时，__name__ 被设为 '__main__'，条件成立，执行 main()。
#   当文件被 `import train_ppo` 导入为模块时，__name__ 是 'train_ppo'，条件不成立，main() 不执行。
#   这种设计让同一个 .py 文件既能作为独立脚本运行，也能被其他文件安全导入。
if __name__ == '__main__':
    main()
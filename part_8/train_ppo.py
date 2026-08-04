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
#   1. 批次大小很小（默认 4），适合单卡调试和教学演示
#   2. RM 奖励为稀疏奖励（仅序列末尾有标量值），但 KL 惩罚提供了逐 token 的稠密信号
#   3. 优势估计使用完整的 GAE 时序差分展开（从序列末尾反向递推累积未来奖励，
#      --gamma 管”多远”、--lam 管”多信”），与 InstructGPT 的标准做法一致
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

        # ─── 预分配四个张量缓存：把变长序列"矩形化"成规整批次 ───
        # 为什么必须预分配？
        #   1. 批处理硬性要求：PyTorch 一次前向只能吃规整矩形 (B, max_len)，
        #      但批次内每条序列长度不同，必须统一到 max_len 再送入模型。
        #   2. 效率：torch.zeros 一次性分配整块显存，循环里只需"按位置填格子"；
        #      若改用循环中逐个 .cat() 拼凑，每次都要重新分配内存 + 拷贝，
        #      时间开销 O(B×L)，还会造成显存碎片。
        #   类比：预分配 = 一张固定大小的画布上按坐标作画；cat = 画一张纸再粘一张。
        # 为什么初始化为全 0？
        #   对每个张量而言，"全 0"恰好是"此处无内容"的语义默认值：
        #   未填的位置、未标记的位置、未收到奖励的位置，都天然有正确含义。
        seq     = torch.zeros(B, max_len, dtype=torch.long, device=device)       # Token 序列张量, 形状 (B, max_len)
        #   ① seq：整个 batch 对齐后的完整 token 序列（prompt + response）的"画布"。
        #      0 是初始占位值（token ID 0），随后所有位置都会被显式覆盖或 padding。
        mask    = torch.zeros(B, max_len, dtype=torch.bool, device=device)      # Response 动作掩码, 形状 (B, max_len)
        #   ② mask：动作掩码——标记哪些 token 是"模型自己生成的 response 部分"。
        #      mask[b, t] = True 表示位置 t 是策略的动作；False = prompt（环境给定，不是策略的选择）。
        #      只有 True 的位置才参与 PPO 损失计算，prompt 部分绝不参与（否则模型会去"学习"输入本身）。
        last_idx = torch.zeros(B, dtype=torch.long, device=device)              # 每条序列末尾索引, 形状 (B,)
        #   ③ last_idx：每条序列最后一个有效 token 的索引（L-1），即"序列在哪结束"。
        #      形状 (B,) 而非 (B, max_len)——每条序列只需要一个数字。
        #      GAE 会用它识别每条轨迹的终止 action，避免把右侧 padding 当成未来状态。
        rewards  = torch.zeros(B, max_len, dtype=torch.float, device=device)    # 稀疏奖励张量（仅序列结尾有值）, 形状 (B, max_len)
        #   ④ rewards：稀疏奖励张量——把每条样本的"一个"标量 RM 奖励，
        #      安放到该序列最后一个有效 token 的位置 rewards[i, L-1] = r_scalar。
        #      为什么必须摊进 (B, max_len) 而非存 (B,)？因为阶段 D 的即时奖励需要和
        #      logprobs/values 保持逐时间步对齐，之后再用 act_mask 只保留 response action。
        #      标量奖励必须"落位"到矩阵里才能参与逐位置的 GAE 递推。
        #      初始全 0 = 该位置未收到奖励（prompt 位置和 response 中间位置天然无奖励）。

        # data 中每一项都是 (完整 token 序列, 原始 prompt 边界, RM 标量奖励)。
        # 下面的循环把每条变长样本放进同一张 (B, max_len) 的"矩形画布"，
        # 同时重新计算截断后的 response 边界、动作 mask 和终点位置。
        for i, (ids, boundary, r_scalar) in enumerate(data):
            # enumerate 同时给出 batch 行号 i 和该行样本内容；i 用来写入 seq/mask/rewards 的第 i 行。
            # ids 的形状是 (L_full,)，包含 prompt + response 的全部 token。
            L_full = ids.numel()                    # 原始序列长度（可能超过批次统一长度 max_len）
            # max_len 是本批次的统一列数：过长样本必须左截断，较短样本稍后右 padding。
            L = min(L_full, max_len)                # 这条样本在画布中实际保留的有效 token 数
            # 如果 L_full > max_len，drop 就是从序列左侧丢弃的 token 数；否则 drop=0。
            # 这个数量用于把原始 boundary 映射到左截断后的新坐标系。
            drop = L_full - L                       # 左侧被裁切丢弃的 token 数量
            # boundary 原本表示 prompt 结束位置；左侧丢掉 drop 个 token 后，边界也要左移 drop。
            # max(0, ...) 防止截断已经穿过整个 prompt：此时保留下来的 token 都视为 response。
            b = max(0, boundary - drop)             # 截断后 prompt/response 分界线（token 索引）
            # 负切片 ids[-L:] 的含义是"取最后 L 个 token"，而不是取前 L 个 token。
            # 左截断专门保留序列尾部，确保最后生成的 response token（尤其是 terminal token）不被丢掉。
            seq[i, :L] = ids[-L:]                   # 将保留的 token 写入第 i 行的前 L 列
            if L < max_len:
                # seq[i, L:] 是第 i 行从有效长度 L 到 max_len-1 的尾部切片，形状为 (max_len-L,)。
                # 这些位置没有真实 token，只是为了和 batch 中最长样本对齐，所以用 pad ID=2 占位。
                # 右 padding 不改变前 L 个真实 token 的位置；后续 act_mask/last_idx 会排除它们。
                seq[i, L:] = 2  # 将短样本的尾部补齐到 max_len
            # mask[i, b:L] 选择第 i 行、列 b（包含）到 L（不包含）的区间，形状为 (L-b,)。
            # b:L 正好是截断后仍保留的 response token 区间；prompt 和 padding 保持 False。
            # 后面 act_mask = mask[:, 1:] 会再右移一格，因为 action t 预测 token t+1。
            mask[i, b:L] = True
            # 稀疏 RM 奖励属于整段 response，但要落在最后一个有效 token 的位置 L-1。
            # 这样 rewards[:, 1:] 对齐到 transition 轴后，最终 action t=L-2 能拿到这个奖励。
            rewards[i, L-1] = r_scalar
            # last_idx 保存的是"最后 token 的索引"而不是长度；它会在 GAE 中识别每条样本的 terminal 边界。
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
        # policy(seq) 会为输入中的每个 token 位置输出一个 value：
        #   values[:, j] 表示看到 token_j 之后，状态 s_j 的价值 V(s_j)，所以原始形状是 (B, T)。
        # 但语言模型的 action 是"用位置 j 预测下一个 token_{j+1}"，一共只有 T-1 个 transition：
        #   logits[:, 0] → 预测 token_1，logprobs[:, 0] → action 0 的概率
        #   logits[:, 1] → 预测 token_2，logprobs[:, 1] → action 1 的概率
        #   ...
        #   logits[:, T-2] → 预测 token_{T-1}，logprobs[:, T-2] → action T-2 的概率
        # 最后一列 values[:, T-1] 代表"读完最后一个输入 token 后的状态"，但没有 token_T 可供它预测，
        # 因此没有对应的 action logprob；为了让 values 与 pol_lp 在同一条 transition 轴上对齐，
        # 只保留 values[:, 0:T-1]，即 (B, T) → (B, T-1)。
        # 注意：丢弃这一列不等于说这个状态的数学 value 必然为 0；这里只是暂不把它放入 action 对齐张量。
        values = values[:, :-1]

        # 仅筛选动作 (Action Tokens, 即 Response 部分) 的对数概率与价值估计
        # 关键理解：PPO 只优化"策略做出的动作"——也就是模型自己生成的 response token。
        #   Prompt 部分的 token 是环境给定的，不是策略的选择，不该参与 actor 的损失计算。
        #
        # ─── 用一个具体序列理解为什么要 mask[:, 1:] ───
        # 假设 token 位置和内容是：
        #   位置:    0    1    2    3    4
        #   内容:   <s>   P   R1   R2  EOS
        #   mask:    F    F    T    T    T
        # 这里 mask 的每一列对应"token 自己是不是 response"，所以 R1/R2/EOS 的位置为 True。
        # 但语言模型的第 t 列 logits 并不是在评估 token_t，而是在预测下一个 token_{t+1}：
        #   logits[0] → 预测 token_1，logprobs[0] → token_1 的概率
        #   logits[1] → 预测 token_2，logprobs[1] → token_2 的概率
        #   logits[2] → 预测 token_3，logprobs[2] → token_3 的概率
        #   logits[3] → 预测 token_4，logprobs[3] → token_4 的概率
        # 因此 logprobs 的第 t 列必须搭配 mask 的第 t+1 列；mask[:, 1:] 就是把这张 token mask
        # 向左对齐到"预测时间步"。上例中 act_mask 变成 [F, T, T, T]，正好选中 R1/R2/EOS。
        # mask[:, 0] 被丢掉不是丢掉第一个 response，而是因为 token_0（通常是 <s>）从未被预测。
        act_mask = mask[:,1:]  # 形状: (B, T-1)，True 的位置 = "被预测的是否是response token"
        # 布尔索引筛选：act_mask 为 True 的位置被拉平为一维向量
        # .detach() 切断梯度——旧策略的概率在 PPO 更新中作为常数基准，不应回传梯度
        old_logp   = pol_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        ref_logp   = ref_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        old_values = values[act_mask].detach()    # 形状: (N_action_tokens,)

        # ==========================================
        # 阶段 D：KL 惩罚、逐 token 即时奖励与 GAE 优势函数计算
        # ==========================================
        # 这是 RLHF 的"奖励塑造 + 优势估计"核心步骤（InstructGPT 的标准做法），分三步走：
        #
        # 1. 计算逐 token KL 散度（保持二维序列结构，供 GAE 使用）
        # 2. 构造逐 token 即时奖励：每个 token = −kl_coef·KL，末尾额外 + RM 分
        # 3. GAE 反向递推：把未来的奖励按 γλ 衰减往回传，算出每个 token 的优势
        #
        # ─── 1. 计算逐 token 的 KL 散度（二维，保持序列结构）───
        # KL 散度衡量两个概率分布之间的"距离"——分布越接近，KL 越接近 0。
        # 公式：KL(π_old || π_ref) ≈ E_{a~π_old}[log π_old(a|s) - log π_ref(a|s)]
        # 单样本近似：KL ≈ log π_old(a|s) - log π_ref(a|s)，每个 token 是独立离散动作，逐元素相减即可。
        # 形状 (B, T-1)：每个预测位置一个 KL 值（含 prompt 位置，稍后会被 act_mask 过滤）。
        # 注意：严格来说 KL 需要对整个词表求和，但那只在"需要完整分布距离"时才必要，
        #       PPO 中采样估计计算量小、效果足够好（InstructGPT 的做法）。
        kl_full = pol_lp - ref_lp  # 形状: (B, T-1) 逐 token KL（旧策略 vs 参考模型）

        # ─── 2. 构造逐 token 即时奖励序列 r_t ───
        # 标准 RLHF 的奖励结构：
        #   每个 token 的即时奖励 = −kl_coef·KL(π‖π_ref)   （"别跑偏"的稠密惩罚）
        #   最后一个 token 额外 + RM 分                      （"人类偏好"的稀疏信号）
        # 为什么 KL 惩罚要进奖励？Reward Model 只看回答好坏，不关心语言是否自然流畅；
        # 若不惩罚偏离，Policy 会"走火入魔"——生成一堆语法错乱但 RM 喜欢的高分词。
        #
        # 语法：rewards[:, 1:] 把 (B, T) 的稀疏 RM 分对齐到预测时刻轴 (B, T-1)：
        #   预测时刻 t 拿到的奖励 = 被预测 token (t+1) 位置上的奖励。
        # 只保留 response action 的奖励；prompt 和右侧 padding 不是 rollout transition，
        # 不能让它们的 KL/value 信号进入 GAE。
        shaped_full = rewards[:, 1:] - args.kl_coef * kl_full
        r_t = torch.where(act_mask, shaped_full, torch.zeros_like(shaped_full))

        # ==========================================
        # 阶段 D-2：完整版 GAE 优势估计 (Generalized Advantage Estimation)
        # ==========================================
        # 为什么要 GAE？——让"远见"往回传。
        #   如果只看即时奖励（γ=0 的单步近视），中间的 token 学不到
        #   "末尾 RM 高分"的信息；GAE 从序列末尾反向递推，
        #   把未来的奖励按指数衰减往回传，每个 token 都能"闻到"最终得分的味道。
        #
        # 数学核心（两步）：
        #   ① 时序差分误差（"惊喜度"）：δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
        #      实际拿到的（当前奖励 + 未来预估）比我原本预测的价值好多少
        #   ② GAE 累积（"带遗忘的惊喜累积器"）：A_t = δ_t + γλ·A_{t+1}
        #      从序列末尾往前滚，未来每步的惊喜按 (γλ) 指数衰减累加
        #
        # 参数直觉：
        #   γ（gamma）：未来奖励的折扣因子——多久远的奖励还算数（0 = 只看当下）
        #   λ（lam）  ：偏差-方差折中——λ=0 只看一步（低方差高偏差，太信价值网络）；
        #               λ=1 看完整条轨迹（无偏高方差，太信单次采样）；0.95 是工业默认折中
        #
        # 维度说明（都在"预测时刻轴"上，形状 (B, T-1)）：
        #   values[b, t] = 看到 token_0..t 后的状态价值（阶段 C 已对齐好）
        #   r_t[b, t]    = 预测时刻 t 的即时奖励（上面第 2 步算好）
        #   递推只覆盖 response action；每条样本自己的最后一个 action 之后没有未来，
        #   padding 和 prompt 位置都会重置递推状态。
        with torch.no_grad():
            T1 = values.size(1)              # 预测时刻轴长度 = T-1
            adv_full = torch.zeros_like(values)      # (B, T-1) 每个预测时刻的优势，actor loss
            returns_full = torch.zeros_like(values)  # (B, T-1) 价值头的回归目标critic loss
            # values[:, t] 是 action t 的当前状态价值，action t 预测 token t+1。
            # 因此 last_idx（最后一个 token 的位置）对应的最后一个 action 是 last_idx - 1。
            time = torch.arange(T1, device=device).unsqueeze(0)  # (1, T-1)
            nonterminal = (time + 1) < last_idx.unsqueeze(1)      #当前 action 执行后，是否还有下一个状态的 value (B, T-1)
            # action t 的下一个状态价值；最后一个时间步没有可用的 next value。
            next_values = torch.cat(
                [values[:, 1:], values.new_zeros((B, 1))], dim=1
            )
            # 初始 A_next = 0：从每条轨迹的终点开始反向递推。
            A_next = values.new_zeros(B)
            for t in range(T1 - 1, -1, -1):   # t 从最后一个预测时刻倒着滚到 0
                # 为什么从后往前？因为 A_t 依赖 A_{t+1}（未来的惊喜），
                # 必须先把后面的算出来才能算前面的——就像搭积木从顶层往下搭。
                # 每条样本在自己的终止 action 处不 bootstrap；padding 位置也不 bootstrap。
                next_v = torch.where(
                    nonterminal[:, t], next_values[:, t], torch.zeros_like(next_values[:, t])
                )
                # ① TD error：δ_t = 实际拿到的（奖励 + 未来预估）− 我原本预测的价值
                delta = r_t[:, t] + args.gamma * next_v - values[:, t]
                # ② 关键递推：A_t = δ_t + γλ·A_{t+1}
                #    未来每步的惊喜按 γλ 衰减累积——λ 越大"看得越远"
                candidate = delta + args.gamma * args.lam * nonterminal[:, t] * A_next
                # 只有 response action 是有效 transition；处理 padding/prompt 时清零，
                # 防止无效位置的信号泄漏到前面的 response。
                A_next = torch.where(act_mask[:, t], candidate, torch.zeros_like(candidate))
                adv_full[:, t] = A_next
                # 价值头回归目标：returns_t = A_t + V(s_t)
                returns_t = A_next + values[:, t]
                returns_full[:, t] = torch.where(
                    act_mask[:, t], returns_t, torch.zeros_like(returns_t)
                )

        # ─── 3. 提取动作位置并归一化 ───
        # act_mask 布尔索引展平 → 只留 response 预测位置（"漏网之鱼倒进一个筐"）
        # .detach()：GAE 算出的优势是采样时刻的常数基准，不该参与反向传播
        adv_final = adv_full[act_mask].detach()          # 形状: (N_action_tokens,)
        returns_final = returns_full[act_mask].detach()  # 形状: (N_action_tokens,)
        # Z-Score 归一化（零均值 + 单位方差）：PPO 的 clip 范围（默认 ±0.2）是固定值，
        # 优势尺度不归一化时 clip 效果不稳定（有时太松、有时太紧）。
        # 使用 unbiased=False，使只有一个 action token 时 std 仍返回 0 而不是 nan。
        adv_std = adv_final.std(unbiased=False).clamp_min(1e-6)
        adv_final = (adv_final - adv_final.mean()) / adv_std

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
        #   adv_final: GAE 计算出的优势函数（已归一化），正值→增大概率，负值→减小概率
        #   new_values, old_values: 新旧价值估计（old_values 用于计算 value clipping）
        #   returns_final: 目标回报（GAE 的 A_t + V(s_t)），Value Head 的回归目标
        #   clip_ratio=0.2: PPO 的核心参数——限制策略更新幅度不超过 ±20%
        #     如果 ratio ∈ [0.8, 1.2]，直接优化；超出范围则用 clip 截断梯度
        #   vf_coef=0.5: Value Function 损失在总损失中的权重系数
        #   ent_coef=0.0: 熵奖励系数（此处关闭，如需鼓励更多探索可设为正值）
        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv_final, new_values, old_values, returns_final,
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

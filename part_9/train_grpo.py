# ==========================================
# Part 9 核心模块：微型 GRPO (Group Relative Policy Optimization) 训练主循环
# ==========================================
# 职责：用 GRPO 算法对 SFT 模型做强化学习微调（RLHF 第三阶段）。
#       结合冻结的 Reference 模型与 Part 7 奖励模型，对每个 Prompt 采样一组回答，
#       用"组内相对奖励"作为优势信号，纯策略（无价值头）更新 Policy。
#
# ─── GRPO 与 Part 8 PPO 的核心区别 ───
#   PPO（Part 8）：
#     - 需要 Value Head（Critic）估计状态价值 V(s)，Advantage = 奖励 - V(s)
#     - 每一步只采样一个回答；KL 惩罚通过改 reward（shaped reward）实现
#   GRPO（本文件，DeepSeekMath 论文）：
#     - 【去掉价值头】！对每个 Prompt 采样 G 个回答组成"组"
#     - Advantage 用"组内相对奖励"：adv_i = r_i - 组均值（同组回答互相对比）
#     - KL 惩罚直接加到损失函数里（total = L_PPO + kl_coef * KL(π||π_ref)）
#     - 好处：省掉价值网络的训练成本；组内对比天然消除了奖励分布的偏移
#
# 整体流程概览（5 个阶段）：
#   A. 采样生成：每步选 P 个 Prompt，每个生成 G 个回答 → B = P×G 条轨迹
#   B. 张量对齐：左截断 + 右 Padding 成规整批次，构造动作掩码
#   C. 对数概率：算 old_logp / ref_logp（逐 token），并逐 token 估计 KL
#   D. 组内优势：按 prompt 分组，adv = 个体奖励 - 组均值，广播到组内每个 token
#   E. PPO 更新：Policy-Only Clipped Loss + KL 惩罚，反向传播更新参数
# ==========================================

# train_grpo.py
from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue  # we will ignore the value head  ← GRPO 用不到价值头，只取 logits
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts, model_logprobs

# Reward model from Part 7
# 跨模块动态导入：Part 7 与 Part 9 是兄弟目录，需要手动加入模块搜索路径
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from grpo_loss import ppo_policy_only_losses


# ==========================================
# 辅助函数：针对 (Prompt, Response) 文本计算奖励模型标量得分
# ==========================================
# 作用：把一个 (prompt, response) 对送入冻结的 Reward Model，得到标量奖励。
# 这个奖励是 GRPO 组内对比的信号来源——RLHF 中的"环境反馈"。
# 与 Part 8 的 compute_reward 相同，只是接口上接收 response_ids（token 列表）而非文本。
# ==========================================
@torch.no_grad()
# 语法：@torch.no_grad() 打分只需前向推理，不构建计算图，节省显存
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt_text: str, response_ids: list[int], device) -> float:
    # Build full formatted text (as in your PPO)
    # 1. 把 response token 解码为文本
    from part_6.formatters import Example, format_example
    resp_text = tok.decode(response_ids)
    # 2. 拼接完整对话格式（prompt + response），与 RM 训练时的输入格式完全一致（先决条件）
    text = format_example(Example(prompt_text, resp_text))
    # 3. 编码 + 截断到 block_size + 加 batch 维 → 形状 (1, seq_len)
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    # 4. RM 前向，取标量奖励
    r = reward_model(x)
    # [0] 去掉 batch 维，.item() 转为 Python float
    return float(r[0].item())


# ==========================================
# GRPO 训练主函数 (Main Training Loop)
# ==========================================
def main():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/grpo-demo')  # 检查点保存目录
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')       # Policy/Ref 的初始化权重
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')  # 打分裁判
    p.add_argument('--steps', type=int, default=100)  # 总训练步数
    # ─── GRPO 特有的两个参数 ───
    p.add_argument('--batch_prompts', type=int, default=32, help='number of distinct prompts per step (before grouping)')
    #   每步选择的【不同 prompt 数量】P。P 个 prompt 各生成 G 个回答 → 每步 B = P×G 条轨迹
    p.add_argument('--group_size', type=int, default=4, help='completions per prompt')
    #   每个 prompt 的【回答数】G，即"组大小"。G 越大，组内基线越准（但显存线性增长）
    p.add_argument('--block_size', type=int, default=256)   # 序列最大长度（含 prompt + response）
    p.add_argument('--resp_len', type=int, default=64)      # response 最大 token 数
    p.add_argument('--kl_coef', type=float, default=0.01)   # KL 惩罚系数（防 Reward Hacking / 语言退化）
    p.add_argument('--lr', type=float, default=1e-5)        # AdamW 学习率（RL 通常比 SFT 小）
    p.add_argument('--bpe_dir', type=str, default=None)     # BPE 词表目录
    p.add_argument('--cpu', action='store_true')            # 强制 CPU
    args = p.parse_args()

    # 语法：`A if 条件 else B` 三元表达式——有 GPU 且未强制 CPU 则用 cuda，否则用 cpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer：初始化统一分词器（BPE 优先，ByteTokenizer 兜底）
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # ==========================================
    # 2. 加载 SFT 检查点并初始化 Policy 与 Reference 模型
    # ==========================================
    # 设计决策：Policy（可训练）与 Reference（冻结）初始权重相同，都来自 Part 6 的 SFT 模型。
    # Reference 是"不敢偏离太远"的锚点——KL 惩罚保证 Policy 不会为了刷分生成乱码。
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    # 语法：dict.get('key', default) 安全读取配置，旧检查点缺字段时降级默认值
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    # Policy：可训练的 GRPO 策略模型（复用 PolicyWithValue，但只用它的 logits 输出）
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # 用 SFT 权重初始化 LM 部分
    policy.eval()  # 先切 eval 模式：生成阶段禁用 Dropout

    # Reference：冻结的基准模型（参数永不更新）
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    # 逐个参数关闭梯度：Ref 不参与反向传播，省显存且保证是不可变的标尺
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # ==========================================
    # 3. 加载 Part 7 训练好的奖励模型 (Reward Model)
    # ==========================================
    # RM 是"裁判"：GRPO 的组内相对奖励完全依赖它的打分质量
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
                     block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4),
                     n_head=rckpt['config'].get('n_head', 4),
                     n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()  # RM 只做推理打分，冻结

    # ==========================================
    # 4. 初始化优化器
    # ==========================================
    # 只优化 policy.parameters()——Ref 和 RM 都不参与更新。
    # betas=(0.9, 0.999) 是 Adam 标准动量参数。
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # 微型 Prompt 池：16 条内置 prompt 循环使用（教学简化版，真实 RLHF 用数万条）
    prompts_pool = sample_prompts(16)

    step = 0
    pool_idx = 0   # 池内游标：控制每步取哪几条 prompt
    G = args.group_size

    while step < args.steps:
        # ==========================================
        # 阶段 A：选择 Prompts 并采样生成回答（含组结构）
        # ==========================================
        # ----- SELECT PROMPTS -----
        # Choose P prompts, each will yield G completions → B = P*G trajectories
        # 每步选 P 个不同 prompt；游标滑过池尾则回到开头（循环使用）
        P = max(1, args.batch_prompts)
        if pool_idx + P > len(prompts_pool):
            pool_idx = 0
        batch_prompts = prompts_pool[pool_idx: pool_idx + P]
        pool_idx += P

        # Tokenize prompt-only texts
        # 格式化 prompt（套模板、去 </s> 让模型续写）并编码为 token ID
        prompt_texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        prompt_in_ids = [tok.encode(t) for t in prompt_texts]

        # ----- GENERATE G COMPLETIONS PER PROMPT -----
        # We will collect all trajectories flat, but track their group/prompt ids.
        # 注意：所有轨迹是"扁平"收集的，但用 prompt_id_of 记住每条轨迹属于哪个组，
        #       后续按组计算相对优势。
        seq_list = []        # list[Tensor of token ids]——完整序列（prompt + response）
        boundary_list = []   # index where response starts in the (possibly clipped) sequence——prompt/response 分界
        prompt_id_of = []    # which prompt this trajectory belongs to (0..P-1)——轨迹归属的组编号
        raw_rewards = []     # scalar reward per trajectory (before KL shaping)——RM 原始奖励（无 KL 塑造）
        last_idx_list = []   # for padding bookkeeping

        with torch.no_grad():
            # 双重循环：外层遍历 P 个 prompt，内层每个 prompt 生成 G 个回答（形成"组"）
            for pid, p_ids in enumerate(prompt_in_ids):
                for g in range(G):
                    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
                    # 关键：temperature=2（高温采样）！
                    # GRPO 需要组内回答【足够多样】，组间对比才有意义；
                    # 如果都用 greedy 解码，G 个回答几乎一样，组内相对奖励就退化成无信号。
                    out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=2, top_k=3)
                    full_ids = out[0].tolist()

                    # split prompt/response
                    # boundary = prompt 截断到上下文窗口后的长度（模型实际看到的 prompt 部分）
                    boundary = len(p_ids[-block_size:])  # prompt length clipped to context
                    resp_ids = full_ids[boundary:]       # 切片出模型新生成的 response 部分
                    # 用 RM 给这条轨迹打分（稀疏标量奖励）
                    r_scalar = compute_reward(rm, tok, batch_prompts[pid], resp_ids, device)

                    # 收集这条轨迹的元数据（注意：boundary 是 prompt 长度，而非 response 边界）
                    seq_list.append(torch.tensor(full_ids, dtype=torch.long))
                    boundary_list.append(boundary)
                    prompt_id_of.append(pid)
                    raw_rewards.append(r_scalar)

        # ==========================================
        # 阶段 B：张量对齐、左截断 Padding 与动作掩码构造
        # ==========================================
        # 核心问题：B 条轨迹长度各不相同，但批处理要求规整矩形。
        # 策略：左截断（丢开头 prompt，保末尾 response——它是要学习的部分）+ 右 Padding。
        # ----- PAD TO BATCH -----
        B = len(seq_list)  # B = P*G
        policy_ctx = getattr(policy, "block_size", block_size)
        # max_len：批次最长序列长度（受上下文窗口上限约束）
        max_len = min(policy_ctx, max(s.numel() for s in seq_list))
        # 预分配三个张量缓存（一次性分配显存，比循环 cat 高效）
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)   # 序列画布（prompt + response）
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)  # 动作掩码（response 位置 = True）
        last_idx = torch.zeros(B, dtype=torch.long, device=device)       # 每条序列末尾索引（预留）

        # keep a per-traj “action positions” mask and response-only boundary
        for i, (ids, bnd) in enumerate(zip(seq_list, boundary_list)):
            L_full = ids.numel()          # 原始序列真实长度
            L = min(L_full, max_len)      # 截断后的有效长度
            drop = L_full - L             # 从左侧丢弃的 token 数
            b = max(0, bnd - drop)  # shifted boundary after left-trim  ← 左截断后重算的 prompt 分界
            seq[i, :L] = ids[-L:]         # 填入截断后的序列（保留末尾 = 保留 response）
            if L < max_len:
                seq[i, L:] = 2  # pad token  ← 尾部补 <pad> 占位
            # actions are predicting token t from <=t-1 → positions [1..L-1]
            # but we only care about response tokens: mask [b..L-1] → actions [b+1..L-1]
            # 动作 = "预测 response 部分 token"；mask 标记 response 的 token 位置 [b..L-1]
            mask[i, b:L] = True
            last_idx[i] = L - 1           # 末尾索引 = 有效长度 - 1（长度 vs 索引的 off-by-one）

        # ==========================================
        # 阶段 C：计算旧策略与参考策略的对数概率及逐 token KL
        # ==========================================
        # ----- LOGPROBS & KL VS REF (token-level) -----
        # model_logprobs returns log p(x[t] | x[:t-1]) for t=1..T-1 over labels=seq[:,1:]
        # model_logprobs 返回 (B, T-1)：位置 t 的预测对应 token_{t+1}（预测者舍尾、被预测者舍头）
        with torch.no_grad():
            pol_lp_full = model_logprobs(policy, seq)  # (B, T-1)——旧策略的对数概率
            ref_lp_full = model_logprobs(ref, seq)     # (B, T-1)——参考模型的对数概率

        # action positions (predict positions [1..T-1]); we want only response tokens:
        # act_mask = mask[:, 1:]：mask 是 token 侧（舍头），与 logprobs 的 (B, T-1) 对齐
        act_mask = mask[:, 1:]  # align to (B, T-1)
        # 布尔索引：mask=True 的位置被展平成一维（"漏网之鱼倒进一个筐"），只留 response 预测
        old_logp = pol_lp_full[act_mask].detach()  # (N_act,)；detach 使旧策略概率成为常数基准
        ref_logp = ref_lp_full[act_mask].detach()  # (N_act,)

        # per-token KL on action tokens
        # 逐 token 的 KL 近似：KL(π_old || π_ref) ≈ log π_old - log π_ref
        # 注意：这个 kl_tok 在当前实现中只用于诊断/参考，实际 KL 惩罚用的是更新后的 kl_now_ref_mean
        kl_tok = (old_logp - ref_logp)  # (N_act,)

        # ==========================================
        # 阶段 D：组内相对优势 (Group Relative Advantage)
        # ==========================================
        # 这是 GRPO 的核心创新，与 PPO 最大的不同：
        #   PPO  ：Advantage = 奖励 - 价值头预测的 V(s)
        #   GRPO ：Advantage = 个体奖励 - 【同组均值】（没有价值头！）
        # 直觉：同一个 prompt 的 G 个回答相互比较——"你这组里最好的那个就值得学"。
        #       组均值充当基线，自动抵消奖励模型的系统性偏差（对某些 prompt 偏好打分偏高/偏低）。
        # ----- SHAPED TRAJECTORY REWARD & GROUP BASELINE -----
        # For GRPO, advantage is trajectory-level and broadcast to its tokens.
        # We include KL shaping at trajectory level using mean token KL per trajectory.
        # First, compute mean KL per trajectory on its action tokens.
        # Build an index map from flat action tokens back to traj ids.
        # We can reconstruct counts by iterating rows.
        # 第一步：建立"扁平动作 token → 所属轨迹"的索引映射。
        # 因为 act_mask 把 token 展平了，后面广播组优势时要知道每个 token 属于哪条轨迹。
        traj_id_for_token = []
        counts = torch.zeros(B, dtype=torch.long, device=device)
        offset = 0
        for i in range(B):
            mrow = act_mask[i]                      # 第 i 条轨迹的动作掩码
            n_i = int(mrow.sum().item())            # 该轨迹的动作 token 数
            if n_i > 0:
                traj_id_for_token.extend([i] * n_i) # 每个动作 token 记下所属轨迹 id i
            counts[i] = n_i
            offset += n_i
        traj_id_for_token = torch.tensor(traj_id_for_token, dtype=torch.long, device=device)  # (N_act,)
        raw_rewards_t = torch.tensor(raw_rewards, dtype=torch.float, device=device)  # (B,) 每条轨迹的 RM 奖励

        # 第二步：计算每个 prompt 组的奖励均值（组基线）
        # 语法：列表推导 [i for i in range(B) if prompt_id_of[i] == pid] 找出属于组 pid 的所有轨迹
        group_mean = torch.zeros(B, dtype=torch.float, device=device)
        for pid in range(P):
            idxs = [i for i in range(B) if prompt_id_of[i] == pid]
            if not idxs:
                continue
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            mean_val = raw_rewards_t[idxs_t].mean()  # 该组 G 个回答的平均奖励
            group_mean[idxs_t] = mean_val            # 组内所有轨迹共享同一个基线值

        # 第三步：轨迹级优势 = 个体奖励 - 组均值（正值 = 该回答比同组平均水平好）
        # Advantage per trajectory, broadcast to its action tokens
        traj_adv = raw_rewards_t - group_mean  # (B,)

        # 第四步：把轨迹级优势广播（复制）到该轨迹的每个动作 token 上
        # 用刚才建立的索引映射：traj_adv[traj_id_for_token] 形状 (N_act,)
        # Build a flat tensor of advantages aligned with old_logp/new_logp on action tokens
        if kl_tok.numel() > 0:
            adv_flat = traj_adv[traj_id_for_token]
        else:
            adv_flat = torch.zeros(0, dtype=torch.float, device=device)  # 空批次保护

        # 第五步：整批优势 Z-Score 归一化（零均值 + 单位方差）
        # 为什么：PPO 的 clip 范围（±0.2）是固定值，优势尺度不归一化时 clip 效果不稳定
        # 语法：.std().clamp_min(1e-6) 防止所有优势相同时除零
        # Normalize advantages (optional but usually helpful)
        if adv_flat.numel() > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std().clamp_min(1e-6))

        # ==========================================
        # 阶段 E：GRPO 损失计算与梯度更新（Policy-Only）
        # ==========================================
        # 与 PPO 不同：GRPO 没有价值损失（没有 value head 可训练），
        # 只有策略损失 + KL 惩罚。整个反向传播只更新 policy 参数。
        # ----- UPDATE (policy-only PPO clipped objective) -----
        policy.train()  # 切换训练模式（启用 Dropout）
        # 前向：只取 logits，忽略 value head 输出（GRPO 不需要价值）
        logits_new, _, _ = policy(seq, None)  # ignore value head
        # 与 Part 8 相同的 logprob 提取流程：log_softmax → 舍尾对齐 → gather 提取
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)  # (B, T-1, V)，预测者舍尾
        labels = seq[:, 1:]                                           # (B, T-1)，被预测者舍头
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        new_logp = new_logp_all[act_mask]                             # (N_act,) 动作 token 上的新 logp

        # 更新后策略与 Reference 的平均 KL（用作损失里的惩罚项）
        # 注意与 kl_tok 的区别：kl_tok 是旧的 (old vs ref)，这里用新的 (new vs ref)
        # Mean KL over action tokens
        kl_now_ref_mean = (new_logp - ref_logp).mean() if new_logp.numel() > 0 else torch.tensor(0.0, device=device)

        # 调用 GRPO 损失函数：policy-only clipped loss + kl_coef × KL(π_new || π_ref)
        # 参数说明：
        #   new_logp/old_logp : 新旧策略对数概率（算重要性比率 ratio）
        #   adv               : 组内相对优势（已归一化）
        #   clip_ratio=0.2    : 单步更新幅度限制 ±20%（PPO 信任域）
        #   ent_coef=0.0      : 熵奖励关闭（GRPO 论文原始做法）
        #   kl_coef           : KL 惩罚强度（防止偏离 Reference 太远）
        out_loss = ppo_policy_only_losses(
            new_logp=new_logp,
            old_logp=old_logp,
            adv=adv_flat,
            clip_ratio=0.2,
            ent_coef=0.0,  # set >0 if you want entropy bonus from -new_logp mean
            kl_coef=args.kl_coef,
            kl_mean=kl_now_ref_mean,
        )
        loss = out_loss.total_loss

        # 标准 PyTorch 训练三步曲：清零梯度 → 反向传播 → 更新参数
        # 语法：opt.zero_grad(set_to_none=True) 把梯度置 None（比置零更省显存）
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：RL 信号噪声大（奖励来自另一个模型），clip 到 1.0 防梯度爆炸（安全阀）
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()  # 切回 eval 模式，为下一轮采样做准备

        # ==========================================
        # 阶段 F：KL 偏移监控与指标日志
        # ==========================================
        # RLHF 训练不可见——loss 下降不代表模型变好，必须监控 KL 指标：
        #   KL_move：本次更新跑了多远（过大→学习率太大；过小→没学到东西）
        #   KL_ref ：累计偏离 SFT 初始模型多远（过大→语言可能退化）
        # Some quick diagnostics (movement vs old, and now vs ref)
        with torch.no_grad():
            lp_post = model_logprobs(policy, seq)[act_mask]  # 更新后用新参数重算 logp
            # 近似 KL(old||new) = E[log π_old - log π_new]：单步更新幅度
            kl_move = (old_logp - lp_post).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)
            # KL(now || ref)：与冻结基准的总偏离
            kl_ref_now = (lp_post - ref_logp).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)

        step += 1
        if step % 10 == 0:
            # 每 10 步打印一次关键指标（f-string 格式化保留小数位便于对比）
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| KL_move {kl_move.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )

    # ==========================================
    # 5. 保存 GRPO 微调后的最终模型检查点
    # ==========================================
    # 保存格式与 Part 6 一致：{'model', 'config'} 两个键，后续可被 eval_ppo.py 加载评估
    # 语法：Path.mkdir(parents=True, exist_ok=True) 递归创建目录，存在也不报错
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved GRPO policy to {args.out}/model_last.pt")


# ==========================================
# 脚本入口：Python 标准守卫模式
# ==========================================
# 语法：if __name__ == '__main__': —— 直接运行时执行 main()，被 import 时不触发
#       让同一个文件既能当脚本跑、也能被安全导入
if __name__ == '__main__':
    main()

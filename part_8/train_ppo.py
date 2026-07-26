# ==========================================
# Part 8 核心模块：微型 PPO (Proximal Policy Optimization) 强化学习微调主循环
# 职责：结合预训练/SFT策略模型、参考模型 (Reference Model) 和 Part 7 奖励模型 (Reward Model)，
#       在单卡上通过在线采样 (On-Policy Rollout) 生成回答、计算 Bradley-Terry/RM 标量奖励并应用 KL 惩罚，
#       最后利用 PPO 剪切损失算法对 Policy 网络的 LM 语言模型及 Value Head 价值头进行联合更新。
# ==========================================

from __future__ import annotations
import argparse, torch
from pathlib import Path

# import torch
# torch.manual_seed(0)  # 保持注释，如需复现可取消注释以固定随机种子

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels
from rollout import model_logprobs

# ─── 跨模块导入 Part 7 的奖励模型 (Reward Model) ───
# 语法：sys.path.append(...) 动态将父目录中的 part_7 路径添加到 Python 模块搜索路径列表中，
# 使得当前脚本能够成功 import part_7/model_reward.py 中定义的 RewardModel 类。
# 语法：Path(__file__).resolve().parents[1] 获取当前文件上上级路径（即项目根目录），拼接 'part_7'
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402  # 语法：noqa: E402 告知 flake8 忽略“import 未放在文件最顶部”的 PEP8 警告

from ppo_loss import ppo_losses


# ==========================================
# 辅助函数：针对 (Prompt, Response) 文本计算奖励模型标量得分
# ==========================================
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    # 1. 格式化样本：将 prompt 与 response 拼接为符合 Part 6/7 规范的 Example 格式文本
    # 语法：__import__('part_6.formatters', fromlist=['Example']) 动态导入 part_6 的 Example 类
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))
    
    # 2. Token 编码与截断：转化为 token ID 序列，并截断至模型支持的最大 block_size 长度
    ids = tok.encode(text)
    
    # 3. 构造输入张量：加上 batch 维度，形状变为 (1, seq_len)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    
    # 4. 前向传播：在 no_grad 模式下调用奖励模型，获得标量 reward 输出，并转换为 Python float 浮点数
    # 语法：with torch.no_grad(): 不保留计算图梯度，减少显存开销并加快推理
    with torch.no_grad():
        r = reward_model(x)
    return float(r[0].item())


# ==========================================
# PPO 训练主函数 (Main Training Loop)
# ==========================================
def main():
    # ─── 1. 命令行参数解析 ───
    # 针对 PPO 训练流程中的关键超参数（学习率、折扣因子 gamma、GAE lambda、KL 系数等）进行解析设置
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo', help="模型检查点保存目录")
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100, help="总训练步数")
    p.add_argument('--batch_size', type=int, default=4, help="每步采样的 Prompt 批次大小")
    p.add_argument('--block_size', type=int, default=256, help="Transformer 序列最大长度上限")
    p.add_argument('--resp_len', type=int, default=64, help="Policy 自动生成 Response 的最大 token 数")
    p.add_argument('--kl_coef', type=float, default=0.01, help="KL 散度惩罚系数，防止 Policy 偏离 Ref 策略过远")
    p.add_argument('--gamma', type=float, default=1.0, help="强化学习奖励折扣因子")
    p.add_argument('--lam', type=float, default=0.95, help="GAE (Generalized Advantage Estimation) 折扣参数")
    p.add_argument('--lr', type=float, default=1e-5, help="AdamW 优化器学习率")
    p.add_argument('--bpe_dir', type=str, default=None, help="BPE 分词器目录路径")
    p.add_argument('--cpu', action='store_true', help="是否强制使用 CPU 进行训练")
    args = p.parse_args()

    # 语法：三元表达式判断硬件设备，自动选择 GPU(cuda) 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer：初始化 RLHF 专用分词器
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # ─── 2. 加载 SFT 检查点并初始化 Policy 与 Reference 模型 ───
    # 语法：torch.load(..., map_location=device) 将保存的模型权重直接加载至指定的硬件设备（CPU/GPU）上
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    # 2.1 初始化可优化的 Policy 模型（包含语言模型 LM 与价值头 Value Head）
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # 使用 Part 6 SFT 权重初始化 LM 部分

    # 2.2 初始化冻结的 Reference 模型（基准参考模型）
    # 设计决策：RLHF 过程中模型极易通过生成语法异常但高分的文本来“欺骗”奖励模型（Reward Hacking）。
    # 因此需要保留一个完全冻结的最初 SFT 模型作为 reference，在 loss 中惩罚与 reference 的 KL 散度。
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)  # 冻结参数，不参与反向传播计算梯度
    ref.eval()  # 设为评估模式，关闭 Dropout

    # ─── 3. 加载 Part 7 训练好的奖励模型 (Reward Model) ───
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()  # 奖励模型仅作为固定评估器，冻结参数并开启 eval 模式

    # ─── 4. 初始化 AdamW 优化器 ───
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ─── 5. 构建微型 Prompt 提示池 ───
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        # ==========================================
        # 阶段 A：在线采样批次收集 (Rollout Generation Batch)
        # ==========================================
        # 1. 循环切片选择当前 Batch 的 Prompt 文本
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts) ]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        # 2. 调用当前 Policy 模型自回归生成 Response 序列
        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)  # 形状: (1, prompt_len)
                # 使用 top_k 和采样温度控制生成随机性与质量
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)
                out_ids.append(out[0].tolist())  # 取出生成的完整 token ID 列表

        # 3. 划分 Prompt 与 Response 边界，并利用 Reward Model 计算环境标量奖励
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            # 根据原始 prompt 的 token 长度（截断至 block_size 范围内）确定生成回答的起始边界位置
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]  # 切片获取生成的 response 部分 ID
            
            # 将生成的 response 解码为文本，并通过 Reward Model 评分得到标量奖励 r_scalar
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        # ==========================================
        # 阶段 B：张量对齐、Padding 填充与动作掩码构造
        # ==========================================
        # 确定批次内最大序列长度 max_len（受限于 Policy 的上下文窗口 block_size）
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(t[0].numel() for t in data))
        B = len(data)
        
        # 预分配张量缓存，避免循环拼接导致频繁内存重新分配
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)       # Token 序列张量, 形状 (B, max_len)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)      # Response 动作掩码, 形状 (B, max_len)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)          # 序列末尾索引, 形状 (B,)
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)  # 稀疏奖励张量（仅序列结尾有值）, 形状 (B, max_len)

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L                 # 从左侧超长裁切掉的 token 数量
            b = max(0, boundary - drop)       # 左裁切后调整后的 prompt/response 边界索引
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # 语法：将短于 max_len 的尾部位置填充为 <pad> token（ID=2）
            mask[i, b:L] = True  # 仅将 response 对应的 token 位置标记为 True（参与 RL 损失计算）
            rewards[i, L-1] = r_scalar  # 标量奖励放在回答序列的最后一个 token 位置上（稀疏奖励）
            last_idx[i] = L-1

        # ==========================================
        # 阶段 C：计算旧策略 (Old Policy) 与参考策略 (Ref) 的对数概率与状态价值
        # ==========================================
        # 语法：model_logprobs 返回 (B, T-1) 形状的张量，表示预测第 t 个 token 的条件对数概率 log p(x_t | x_<t)
        pol_lp = model_logprobs(policy, seq)
        ref_lp = model_logprobs(ref, seq)
        
        # 前向传播计算序列中各个位置的状态价值 (Value Estimates)
        # 语法：logits, values, _ = policy(seq, None) 接收 (B, T) 输入，输出 values 形状为 (B, T)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        values = values[:, :-1]  # 切片丢弃最后一个预测位置，使得 values 形状 (B, T-1) 与 pol_lp 对齐

        # 仅筛选动作 (Action Tokens, 即 Response 部分) 的对数概率与价值估计
        # 语法：act_mask = mask[:, 1:] 切片避开第 0 个 token，因 logprobs 预测的是 t+1 位置
        act_mask = mask[:,1:]  
        old_logp = pol_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        ref_logp = ref_lp[act_mask].detach()    # 形状: (N_action_tokens,)
        old_values = values[act_mask].detach()  # 形状: (N_action_tokens,)

        # ==========================================
        # 阶段 D：KL 惩罚、塑造奖励 (Shaped Rewards) 与优势函数 (Advantage) 计算
        # ==========================================
        # 1. 计算每个动作 token 的 KL 散度近似值：KL ≈ log π_old(a|s) - log π_ref(a|s)
        kl = (old_logp - ref_logp)
        
        # 2. 塑造奖励 (Shaped Reward)：在原始 RM 标量奖励的基础上，减去偏移 KL 散度带来的惩罚项
        # 避免模型为了追求 RM 高分而疯狂生成背离原始语言模型的乱码
        shaped_r = rewards[:,1:][act_mask] - args.kl_coef * kl

        # 3. 优势函数 (Advantage) 与 目标回报 (Returns) 估计：
        # 教程简化版实现：将即时 shaped 奖励视作目标回报 returns，优势 adv = returns - old_values
        returns = shaped_r
        adv = returns - old_values
        
        # 语法：(adv - adv.mean()) / (adv.std().clamp_min(1e-6)) 对优势函数做 Batch 归一化，
        # 稳定 PPO 梯度更新幅度。clamp_min(1e-6) 防止除以 0 导致数值 NaN 溢出。
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

        # ==========================================
        # 阶段 E：PPO 损失计算与梯度更新 (PPO Update Pass)
        # ==========================================
        policy.train()  # 切换至训练模式
        
        # 前向传播重新计算当前策略网络在新参数下的对数概率与价值输出
        logits_new, values_new_full, _ = policy(seq, None)
        # 语法：torch.log_softmax(..., dim=-1) 在词表维度计算 Log-Softmax，稳定对数概率计算
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:,1:]
        
        # 语法：.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        #   labels.unsqueeze(-1) 将标签扩展为 (B, T-1, 1)
        #   .gather(-1, ...) 从词表维度中精确提取目标标签对应 token 的对数概率
        #   .squeeze(-1) 恢复为 (B, T-1) 形状
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        
        # 提取 Action 掩码位置的最新对数概率与价值估计
        new_logp = new_logp_all[act_mask]
        new_values = values_new_full[:, :-1][act_mask]

        # 调用 ppo_losses 核心算法计算 PPO Clipped Loss、Value Loss 与 Total Loss
        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss

        # 语法：opt.zero_grad(set_to_none=True) 将梯度清空设为 None（相比零张量清空能稍微节省显存并提速）
        opt.zero_grad(set_to_none=True)
        loss.backward()  # 反向传播计算梯度
        
        # 语法：clip_grad_norm_ 将梯度范数裁剪至最大 1.0，防止强化学习更新过程中出现梯度爆炸
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()  # 更新策略模型参数
        policy.eval()  # 切回评估模式

        # ==========================================
        # 阶段 F：KL 偏移监控与指标日志记录
        # ==========================================
        with torch.no_grad():
            # 1. KL(old || new)：监控一次 PPO 更新后，策略相比采样策略偏离了多少
            lp_post = model_logprobs(policy, seq)          # 形状: (B, T-1)
            lp_post = lp_post[act_mask]                    # 仅选动作位置
            kl_post = (old_logp - lp_post).mean()          # 近似 E[log π_old - log π_new]

            # 2. KL(now || ref)：监控当前更新后的策略与冻结基准 reference 模型之间的总偏离距离
            lp_now = lp_post
            kl_ref_now = (lp_now - ref_logp).mean()        # 近似 E[log π_now - log π_ref]

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| value loss {out_loss.value_loss.item():.4f} | KL_move {kl_post.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )

    # ─── 6. 保存 PPO 微调后的最终模型检查点 ───
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved PPO policy to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()
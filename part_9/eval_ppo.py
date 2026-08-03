# ==========================================
# Part 9：GRPO 策略模型评估脚本 (GRPO Policy Evaluator)
# ==========================================
# 功能：加载 Part 9 训练好的 GRPO 策略模型 (Policy) 与 Part 6 的 SFT 参考模型 (Reference)，
#       在测试 Prompt 提示词集上自回归生成回答，通过 Part 7 奖励模型打分，
#       评估 GRPO 强化学习微调对回答偏好对齐带来的提升。
#
# 与 Part 8 的 eval_ppo.py 结构一致，只是评估对象换成了 GRPO 训练出的 checkpoint。
# 评估哲学：对比 GRPO 模型与 SFT 初始模型的平均奖励，判断 RLHF 训练是否真正带来提升。
# ==========================================

from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# 跨模块动态导入 Part 7 的奖励模型 (Reward Model)
# 语法：sys.path.append(...) 将 part_7 目录加入模块搜索路径（Part 7 与 Part 9 是兄弟目录）
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402  # 语法：noqa: E402 忽略"import 未在文件顶部"的 PEP8 警告


# ==========================================
# 策略评估核心函数 (Score Policy Function)
# ==========================================
def score_policy(policy_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n: int = 16):
    """在测试 Prompt 提示词集上自回归生成回答，并利用 Reward Model 计算平均奖励得分。

    参数说明:
        policy_ckpt : GRPO 训练后保存的 Policy 权重路径
        rm_ckpt     : Part 7 训练好的 Reward Model 权重路径
        bpe_dir     : 可选的 BPE 分词器目录（必须与训练时词表一致）
        n           : 评估使用的 Prompt 数量（默认 16，速度与统计稳定性折中）
    """
    # 推理设备：有 GPU 用 cuda，否则回退 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化统一分词器，block_size=256 与训练一致
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # ─── 重建并加载待评估的 GRPO 策略模型 ───
    # 语法：torch.load(..., map_location=device) 将权重加载到指定设备，防止 CPU/GPU 不匹配报错
    ckpt = torch.load(policy_ckpt, map_location=device)
    # 语法：dict.get('key', default) 安全读取配置；旧检查点缺 config 时降级用默认超参
    cfg = ckpt.get('config', {})
    pol = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    pol.load_state_dict(ckpt['model'])
    # 语法：pol.eval() 切换评估模式——禁用 Dropout 等随机行为，保证生成结果确定可复现
    pol.eval()

    # ─── 重建并加载用于对比的 SFT 参考模型 (Reference Model) ───
    # 评估目的是对比 GRPO 训练前后的提升：Reference = Part 6 SFT 初始模型（冻结）
    ref = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    # 硬编码读取 Part 6 的 SFT 权重路径（假设从项目根目录或 part_9/ 下执行）
    ckpt_ref = torch.load("../part_6/runs/sft-demo/model_last.pt", map_location=device) # hardcoded path to SFT checkpoint
    # 注意：加载到 ref.lm（底层语言模型）而非整体——Part 6 检查点没有 value head 权重
    ref.lm.load_state_dict(ckpt_ref['model'])
    # 冻结参考模型参数：节省显存 + 保证公平对比（它是不变的标尺）
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # ─── 重建并加载 Part 7 奖励模型 (Reward Model) ───
    # RM 是 RLHF 的"裁判"：对 (Prompt, Response) 对输出标量分数，分数越高越符合人类偏好
    rckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    # ─── 在 Prompt 测试集中逐条采样生成与打分 ───
    prompts = sample_prompts(n)
    rewards = []  # 记录每条样本上 GRPO 模型生成回答的 RM 奖励

    for p in prompts:
        # 格式化 Prompt：套上对话模板并移除终止符 </s>（让模型"继续写"而不是停在这里）
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        # 语法：ids[-tok.block_size:] 右对齐截断到上下文窗口；外层 [] 加 batch 维，形状 (1, prompt_len)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        # 语法：with torch.no_grad() 推理阶段关闭自动求导，节省显存加速生成
        with torch.no_grad():
            # 两个模型用完全相同的生成参数（控制变量法），唯一的变量是模型权重
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)      # GRPO 策略
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)  # SFT 参考
        # 提取 Response：完整序列 = prompt + response，用 prompt 长度切片分离出 response 部分
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])
        resp_old = tok.decode(y_old[0].tolist()[len(ids[-tok.block_size:]):])

        # 拼接 (Prompt, Response) 完整对话格式，与 RM 训练时格式完全对齐（否则打分不准）
        from part_6.formatters import Example, format_example
        text = format_example(Example(p, resp))
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)
        with torch.no_grad():
            # rm(z) 返回形状 (1,) 的标量张量，[0].item() 取出第 0 个元素并转为 Python float
            r = rm(z)[0].item()
        rewards.append(r)
    # 平均奖励：max(1, len) 防止空列表除零（极端情况的边界保护）
    return sum(rewards)/max(1,len(rewards))


# ==========================================
# 命令行入口脚本 (CLI Main Entry)
# ==========================================
# 语法：if __name__ == '__main__': 入口保护——直接运行时执行评估，被 import 时不触发
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # 必填参数：GRPO 策略 checkpoint 与 RM checkpoint 路径
    p.add_argument('--policy_ckpt', type=str, required=True)
    p.add_argument('--reward_ckpt', type=str, required=True)
    # 可选参数：数据划分标记（微型脚本中保留以维持接口兼容，实际未使用）
    p.add_argument('--split', type=str, default='val[:32]')  # unused in this tiny script
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    # 执行评估并输出平均奖励，f"{avg_r:.4f}" 保留 4 位小数便于版本间对比
    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.bpe_dir, n=16)
    print(f"Avg RM reward: {avg_r:.4f}")

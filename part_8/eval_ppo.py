# ==========================================
# Part 8：PPO 策略模型评估脚本 (PPO Policy Evaluator)
#
# 功能：加载在 Part 8 训练好的 PPO 策略模型 (Policy)、Part 6 初始的 SFT 参考模型 (Reference)，
#       以及 Part 7 训练好的奖励模型 (Reward Model)。在测试 Prompt 提示词集上自回归生成回答，
#       通过奖励模型对生成质量进行标量打分，评估 RLHF (PPO) 强化学习微调对模型回答偏好对齐带来的提升。
#
# 整体数据流与控制流：
#   测试 Prompt 字符串 ──> 格式化为 Prompt 前缀 ──> Tokenizer 编码为 token ID 张量 (1, T_prompt)
#   ──> PPO Policy 与 SFT Ref 模型并行自回归生成 ──> 切片分离出生成回答 Response 字符串
#   ──> 拼接为 (Prompt, Response) 对 ──> Reward Model 前向推理 ──> 获得标量得分 Reward Scalar
#   ──> 统计平均 Reward 并输出评估报告
# ==========================================

from __future__ import annotations

# argparse：构建标准命令行接口 (CLI)，方便在终端传入不同的权重 checkpoint 路径与参数
# torch：提供张量计算、模型权重加载 (torch.load) 及设备管理 (CPU/GPU)
import argparse, torch
from pathlib import Path

# PolicyWithValue：包含 Transformer 语言模型 (Actor) 与价值头 (Critic) 的联合策略网络
from policy import PolicyWithValue
# RLHFTokenizer：统一分词器（支持 BPE / ByteTokenizer 降级回退）
# sample_prompts：获取 Alpaca 或内置微型 Prompt 测试集
# format_prompt_only：将原始输入 Prompt 格式化为标准对话模板
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# ─── 跨模块动态导入 Part 7 的奖励模型 (Reward Model) ───
# 语法：sys.path.append(...) 动态将父目录中的 part_7 路径添加到 Python 模块搜索路径中。
# 语法：Path(__file__).resolve().parents[1] 获取当前文件上上级路径（即项目根目录），拼接 'part_7'，
# 保证无论从哪个目录下执行 `python part_8/eval_ppo.py`，都能精确找到 part_7/model_reward.py。
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402  # 语法：noqa: E402 告知 linter 忽略“import 未在文件最顶部”的 PEP8 警告


# ==========================================
# 策略评估核心函数 (Score Policy Function)
# ==========================================
def score_policy(policy_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n: int = 16) -> float:
    """在测试 Prompt 提示词集上自回归生成回答，并利用 Reward Model 计算平均奖励得分。

    参数说明:
        policy_ckpt : PPO 微调后保存的 Policy 权重路径（如 runs/ppo-demo/model_last.pt）
        rm_ckpt     : Part 7 训练好的 Reward Model 权重路径（如 ../part_7/runs/rm-demo/model_last.pt）
        bpe_dir     : 可选的 BPE 分词器目录路径（必须与训练时使用的词表保持一致）
        n           : 评估使用的 Prompt 测试样本数量（默认 16 条）
    """
    # ─── 1. 推理硬件设备与分词器初始化 ───
    # 语法：torch.cuda.is_available() 自动检测是否有可用 GPU，有则优先使用 'cuda' 提速推理，否则使用 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 RLHF 统一分词器，设定最大上下文窗口大小为 256；若指定了 bpe_dir 则从磁盘加载预训练词表
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # ─── 2. 重建并加载待评估的 PPO 策略模型 (Policy Model) ───
    # 语法：torch.load(..., map_location=device) 将权重直接加载至指定设备，防止在 CPU 环境下加载 GPU 检查点报错
    ckpt = torch.load(policy_ckpt, map_location=device)
    
    # 语法：dict.get('key', default) 提取检查点中的模型配置参数；若旧检查点未保存 config 则降级使用默认参数
    cfg = ckpt.get('config', {})
    
    # 按照检查点保存的超参数重建 PolicyWithValue 网络结构（确保词表大小、隐维、层数、头数与训练时完全一致）
    pol = PolicyWithValue(
        vocab_size=cfg.get('vocab_size', tok.vocab_size),
        block_size=cfg.get('block_size', tok.block_size),
        n_layer=cfg.get('n_layer', 2),
        n_head=cfg.get('n_head', 2),
        n_embd=cfg.get('n_embd', 128)
    ).to(device)
    
    # 将加载的参数字典写入策略模型
    pol.load_state_dict(ckpt['model'])
    
    # 语法：pol.eval() 切换策略模型至评估模式，禁用 Dropout 等随机行为，确保生成结果确定稳定
    pol.eval()

    # ─── 3. 重建并加载用于对比的初始 SFT 参考模型 (Reference Model) ───
    # 评估的目的不仅是看 PPO 模型得分，还要对比其相比 PPO 训练前（即 Part 6 SFT 阶段）是否有提升。
    ref = PolicyWithValue(
        vocab_size=cfg.get('vocab_size', tok.vocab_size),
        block_size=cfg.get('block_size', tok.block_size),
        n_layer=cfg.get('n_layer', 2),
        n_head=cfg.get('n_head', 2),
        n_embd=cfg.get('n_embd', 128)
    ).to(device)
    
    # 硬编码读取 Part 6 的 SFT 模型权重检查点路径
    ckpt_ref = torch.load("../part_6/runs/sft-demo/model_last.pt", map_location=device)
    ref.lm.load_state_dict(ckpt_ref['model'])
    
    # 语法：requires_grad_(False) 显式冻结参考模型的所有参数梯度，节省显存开销并防止误操作修改权重
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # ─── 4. 重建并加载 Part 7 奖励模型 (Reward Model) ───
    # 加载 Part 7 训练好的 RM 检查点
    rckpt = torch.load(rm_ckpt, map_location=device)
    
    # 按照 RM 检查点保存的配置初始化 RewardModel 架构（包含 Transformer 编码器 + 标量输出头）
    rm = RewardModel(
        vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size=rckpt['config'].get('block_size', tok.block_size),
        n_layer=rckpt['config'].get('n_layer', 4),
        n_head=rckpt['config'].get('n_head', 4),
        n_embd=rckpt['config'].get('n_embd', 256)
    ).to(device)
    
    # 写入权重并切换至评估模式
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    # ─── 5. 在 Prompt 测试集中逐条采样生成与打分 ───
    # 从 Alpaca 数据集或内置 Prompt 池中抽取前 n 条 Prompt
    prompts = sample_prompts(n)
    rewards = []  # 记录每条样本在 PPO Policy 生成回答上的 RM 奖励得分

    for p in prompts:
        # 5.1 格式化 Prompt 前缀
        # 语法：format_prompt_only(p) 将原始问题填充进对话模板，replace('</s>', '') 移除终止符，方便模型顺畅续写
        prefix = format_prompt_only(p).replace('</s>', '')
        
        # 将格式化后的 Prompt 转换为 token ID 列表
        ids = tok.encode(prefix)
        
        # 语法：x = torch.tensor([...], dtype=torch.long, device=device)
        #   [ids[-tok.block_size:]] 将长度限制在 block_size 范围内（右对齐滑窗），并增加 batch 维度 1，形状变为 (1, prompt_len)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        # 5.2 并行自回归生成回答 (Generation)
        # 语法：with torch.no_grad(): 关闭 PyTorch 的自动求导引擎（Autograd），在推理阶段不构建计算图，大幅降低显存并加速生成
        with torch.no_grad():
            # pol.generate(...)：调用 PPO 策略模型自回归生成，最多生成 128 个新 token
            # 参数说明：temperature=0.2 较低采样温度提高确定性；top_k=50 限制候选采样范围
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
            
            # 参考模型（SFT）也生成回答（供对比参考）
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)

        # 5.3 提取与解码生成回答 (Response Decoding)
        # 语法：y[0].tolist() 取出生成的完整序列 token ID 列表（格式：[Prompt Tokens... + Response Tokens...]）
        # 语法：[len(ids[-tok.block_size:]):] 使用输入 Prompt 的实际 token 长度切片，精确定位并截取出模型新生成的 Response token ID
        resp_token_ids = y[0].tolist()[len(ids[-tok.block_size:]):]
        resp_old_token_ids = y_old[0].tolist()[len(ids[-tok.block_size:]):]

        # 语法：tok.decode(...) 将 token ID 序列重新还原为人类可读的字符串文本
        resp = tok.decode(resp_token_ids)
        resp_old = tok.decode(resp_old_token_ids)

        # 5.4 格式化完整的 (Prompt, Response) 样本并利用 RM 打分
        # 语法：__import__ 方式导入 Part 6 的 Example 与 format_example 工具
        from part_6.formatters import Example, format_example
        
        # 拼接成完整的对话格式文本（如 "Instruction: ...\n\nResponse: ..."），与 Reward Model 训练时的格式完全对齐
        text = format_example(Example(p, resp))
        
        # 编码为 token ID 张量，截断至 block_size，增加 batch 维，形状 (1, seq_len)
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)
        
        # 调用 Reward Model 前向传播计算标量奖励分值
        with torch.no_grad():
            # rm(z) 返回形状为 (1,) 的标量张量，[0].item() 取出标量浮点数值
            r = rm(z)[0].item()
            
        rewards.append(r)

    # ─── 6. 汇总计算平均奖励得分 (Average Reward Calculation) ───
    # 语法：sum(rewards) / max(1, len(rewards)) 计算所有测试 Prompt 的平均 Reward 标量
    # 语法：max(1, len(rewards)) 边界防错保护，防止空列表导致 ZeroDivisionError 除以零异常
    return sum(rewards) / max(1, len(rewards))


# ==========================================
# 命令行入口脚本 (CLI Main Entry)
# ==========================================
# 语法：if __name__ == '__main__': 入口保护，保证脚本直接运行 `python eval_ppo.py` 时执行评估，被 import 时不触发自动运行
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="PPO Policy Reward Score Evaluator")
    
    # 必填参数：PPO 策略模型 checkpoint 路径（如 runs/ppo-demo/model_last.pt）
    p.add_argument('--policy_ckpt', type=str, required=True, help="待评估的 PPO 策略模型检查点路径")
    
    # 必填参数：Part 7 训练好的 Reward Model checkpoint 路径
    p.add_argument('--reward_ckpt', type=str, required=True, help="Part 7 奖励模型检查点路径")
    
    # 可选参数：数据划分标记（教学微型脚本中保留该 CLI 参数以维持接口兼容性）
    p.add_argument('--split', type=str, default='val[:32]', help="微型评估脚本保留的切片参数")
    
    # 可选参数：指定预训练 BPE 词表保存目录路径
    p.add_argument('--bpe_dir', type=str, default=None, help="BPE 分词器目录路径")
    
    args = p.parse_args()

    # 执行评估逻辑并输出测试集上的平均奖励分值
    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.bpe_dir, n=16)
    
    # 语法：f"{avg_r:.4f}" 格式化输出浮点数，保留 4 位小数
    print(f"Avg RM reward: {avg_r:.4f}")
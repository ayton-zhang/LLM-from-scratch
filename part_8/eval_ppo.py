# ==========================================
# Part 8：PPO 策略模型评估脚本 (PPO Policy Evaluator)
# ==========================================
# 功能：加载在 Part 8 训练好的 PPO 策略模型 (Policy)、Part 6 初始的 SFT 参考模型 (Reference)，
#       以及 Part 7 训练好的奖励模型 (Reward Model)。在测试 Prompt 提示词集上自回归生成回答，
#       通过奖励模型对生成质量进行标量打分，评估 RLHF (PPO) 强化学习微调对模型回答偏好对齐带来的提升。
#
# 评估的设计哲学——"三模型对比法"：
#   为什么要同时加载 Policy 模型和 Reference 模型？PPO 评估的黄金标准是"相对提升"而非"绝对分数"。
#   通过对比 PPO 模型和 SFT 初始模型的生成结果，可以直观判断 RLHF 训练是否真正带来了偏好对齐的提升，
#   而不仅仅是 Reward Model 的"个人喜好"偏高。这也呼应了 PPO 训练中 KL 惩罚的设计意图——
#   我们希望 Policy 变好，但不要偏离初始的 SFT 能力太远。
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
#   Actor（语言模型）负责生成 Token；Critic（价值头）负责估计状态价值，二者共享同一个 Transformer 主干。
#   评估脚本只用到 Actor 的生成能力，不涉及 Critic 的价值估计。
from policy import PolicyWithValue
# RLHFTokenizer：统一分词器（支持 BPE / ByteTokenizer 降级回退）
# sample_prompts：获取 Alpaca 或内置微型 Prompt 测试集
# format_prompt_only：将原始输入 Prompt 格式化为标准对话模板
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# ─── 跨模块动态导入 Part 7 的奖励模型 (Reward Model) ───
# Part 7 的 RewardModel 位于项目根目录下的 part_7/ 子目录，不在 part_8/ 的包范围内。
# 因此需要动态将 part_7 添加到 Python 模块搜索路径中。
#
# 语法：sys.path.append(...) 动态将父目录中的 part_7 路径添加到 Python 模块搜索路径中。
# 语法：Path(__file__).resolve().parents[1] 获取当前文件上上级路径（即项目根目录），拼接 'part_7'，
# 保证无论从哪个目录下执行 `python part_8/eval_ppo.py`，都能精确找到 part_7/model_reward.py。
#       parents 是一个序列：parents[0] = part_8/, parents[1] = 项目根目录/
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
# 语法：from model_reward import RewardModel 必须在 sys.path.append 之后才能执行，
#       因为在此之前 part_7 还不是 Python 能识别的包路径。
# 语法：noqa: E402 告知 linter（如 flake8）忽略该行的 "module level import not at top of file" 警告。
#       E402 是 flake8 的错误码，表示 import 语句没有放在文件最顶部。
from model_reward import RewardModel  # noqa: E402


# ==========================================
# 策略评估核心函数 (Score Policy Function)
# ==========================================
# 这是整个评估脚本的"心脏"——从模型重建到生成打分，所有逻辑都收敛在此函数中。
# 设计为独立函数而非内联脚本的好处：
#   1. 可被其他模块导入复用（如 CI 流程、自动化评测）
#   2. 参数化方便：只需传入不同的 checkpoint 路径即可对比多次训练的结果
#   3. 职责单一：只做评估，不混入训练逻辑
# ==========================================
def score_policy(policy_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n: int = 16) -> float:
    """在测试 Prompt 提示词集上自回归生成回答，并利用 Reward Model 计算平均奖励得分。

    参数说明:
        policy_ckpt : PPO 微调后保存的 Policy 权重路径（如 runs/ppo-demo/model_last.pt）。
                      检查点内部应包含 'model'（权重 state_dict）和 'config'（超参数字典）两个键。
        rm_ckpt     : Part 7 训练好的 Reward Model 权重路径（如 ../part_7/runs/rm-demo/model_last.pt）。
                      同样需要包含 'model' 和 'config' 键。
        bpe_dir     : 可选的 BPE 分词器目录路径（必须与训练时使用的词表保持一致）。
                      若为 None，分词器回退至 ByteTokenizer，可能导致评估结果与训练时不一致（词表不同）。
        n           : 评估使用的 Prompt 测试样本数量（默认 16 条）。
                      这个数字不宜太大：因为每条 Prompt 都需要完整走一次"生成→RM打分"流程，
                      16 条能在 1-2 分钟内完成评估，同时给出统计上较为稳定的平均分。
    """
    # ─── 1. 推理硬件设备与分词器初始化 ───
    # 语法：torch.cuda.is_available() 自动检测是否有可用 GPU。
    #       CUDA（Compute Unified Device Architecture）是 NVIDIA 的 GPU 并行计算平台，
    #       有 CUDA 就用 GPU 加速推理（速度快 10~100 倍），没有则退回到 CPU（兼容性好但慢）。
    # 语法：torch.device('cuda' if ... else 'cpu') 是 Python 三元表达式，
    #       等价于 if/else 赋值两行的缩写。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 RLHF 统一分词器，设定最大上下文窗口大小为 256；
    # 若指定了 bpe_dir 则从磁盘加载预训练词表（推荐，保证词表与训练时一致），
    # 否则回退至 ByteTokenizer（兜底方案，但可能导致词表不匹配）。
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    # ─── 2. 重建并加载待评估的 PPO 策略模型 (Policy Model) ───
    # 为什么不能直接复用训练时的 model 对象？原因有二：
    #   1. 训练脚本的 model 对象已随进程结束而销毁（Python 对象不持久化到磁盘）
    #   2. 评估时可能使用不同的硬件/环境，重新构建再加载权重是最可靠的恢复方式
    #
    # 重建流程：初始化网络结构（骨架）→ torch.load 读磁盘权重（血肉）→ load_state_dict 填充（注入灵魂）
    #
    # 语法：torch.load(..., map_location=device) 将权重张量直接加载至指定设备。
    #       如果不指定 map_location，GPU 上保存的 checkpoint 在无 GPU 环境下加载会报错；
    #       加上后 PyTorch 自动将权重从 GPU 显存重新映射（remap）到 CPU 内存中。
    ckpt = torch.load(policy_ckpt, map_location=device)

    # 语法：dict.get('key', default) 安全地从字典中提取值。
    #       如果检查点中没有 'config' 键（可能是旧格式的检查点），不会崩，而是返回一个空字典 {}，
    #       后续 cfg.get(...) 会使用各自的默认值（n_layer=2, n_head=2, n_embd=128 等）。
    cfg = ckpt.get('config', {})

    # 按照检查点保存的超参数重建 PolicyWithValue 网络结构。
    # 核心原则：重建时的网络结构必须与保存权重时的结构逐参数一一对应，
    # 否则 load_state_dict 会因形状不匹配而报错（比如训练时 n_embd=256 但这里重建时写了 128）。
    # 使用 cfg.get(key, default) 确保向后兼容没有 config 字段的旧检查点。
    pol = PolicyWithValue(
        vocab_size=cfg.get('vocab_size', tok.vocab_size),
        block_size=cfg.get('block_size', tok.block_size),
        n_layer=cfg.get('n_layer', 2),
        n_head=cfg.get('n_head', 2),
        n_embd=cfg.get('n_embd', 128)
    ).to(device)  # 语法：.to(device) 将模型的所有参数和缓冲区移动到目标设备（GPU 或 CPU）

    # 将加载的参数字典写入策略模型。
    # load_state_dict 要求 checkpoint 中的参数名与模型参数名完全匹配（含嵌套前缀），
    # 因此训练时保存的 ckpt['model'] 结构和 PolicyWithValue 的 state_dict 结构必须一致。
    pol.load_state_dict(ckpt['model'])

    # 语法：pol.eval() 切换策略模型至评估模式。
    #       这行代码的实际效果：禁用 Dropout（训练时随机丢弃神经元，评估时全部保留）、
    #       禁用 BatchNorm 的统计量更新。确保同一输入每次前向传播得到相同的结果（确定性）。
    #       训练时用 model.train()，评估时用 model.eval()，漏掉这行会导致生成结果不可复现。
    pol.eval()

    # ─── 3. 重建并加载用于对比的初始 SFT 参考模型 (Reference Model) ───
    # 评估的目的不仅是看 PPO 模型得分，还要对比其相比 PPO 训练前（即 Part 6 SFT 阶段）是否有提升。
    # SFT（Supervised Fine-Tuning）：在人类编写的"问题-答案"对上进行监督学习微调，让模型学会
    # 按照指令格式生成文本。PPO 在 SFT 的基础上进一步用奖励信号优化，使回答更符合人类偏好。
    #
    # Reference 模型使用与 Policy 相同的架构（同骨架、同词表），但加载的是 Part 6 SFT 的旧权重。
    # 评估脚本会同时让 Policy 和 Reference 生成回答，方便对比 RLHF 训练前后的效果差异。
    ref = PolicyWithValue(
        vocab_size=cfg.get('vocab_size', tok.vocab_size),
        block_size=cfg.get('block_size', tok.block_size),
        n_layer=cfg.get('n_layer', 2),
        n_head=cfg.get('n_head', 2),
        n_embd=cfg.get('n_embd', 128)
    ).to(device)

    # PPO 评估时硬编码读取 Part 6 的 SFT 模型权重检查点路径。
    # 设计说明：这里使用相对于当前工作目录的路径（../part_6/runs/sft-demo/model_last.pt），
    # 假设用户从项目根目录或 part_8/ 下执行脚本。如果路径不对，则需要调整或传入参数。
    ckpt_ref = torch.load("../part_6/runs/sft-demo/model_last.pt", map_location=device)
    # 注意：这里加载的是 ref.lm（底层的语言模型）而非 ref 整体。
    # 因为 Part 6 的检查点只保存了语言模型的权重（没有 PolicyWithValue 的 value head），
    # 所以需要直接填充到 ref.lm（即 PolicyWithValue 内部的 Transformer 部分）。
    # value head 保持随机初始化——评估脚本不需要价值估计，不影响结果。
    ref.lm.load_state_dict(ckpt_ref['model'])

    # 语法：for p_ in ref.parameters(): p_.requires_grad_(False)
    #       .parameters() 返回模型中所有可训练参数的迭代器（包括 Transformer 层和 value head）。
    #       requires_grad_(False) 显式冻结参考模型的所有参数梯度。
    # 为什么要冻结？
    #   1. Reference 模型在 PPO 训练中就是"不可训练的标尺"，评估时也必须保持冻结，确保公平对比
    #   2. 关闭梯度计算后 PyTorch 不会为这些参数分配梯度缓存，节省 GPU 显存
    #   3. 防止误操作——如果有人意外在这个模型上调用 backward()，不会有任何参数被修改
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # ─── 4. 重建并加载 Part 7 奖励模型 (Reward Model) ───
    # Reward Model（奖励模型）是 RLHF 中的"裁判"——它接收一个 (Prompt, Response) 对，
    # 输出一个标量分数，代表这个回答的"人类偏好程度"。分数越高，回答越符合人类期望。
    #
    # Reward Model 的架构：Transformer 编码器（与语言模型相同的主干） + 线性输出头（将
    # 最后一个 token 的隐状态映射为一个标量）。与语言模型的核心区别在于输出：语言模型输出
    # V 维的 Token 概率分布，Reward Model 只输出 1 维的得分标量。
    #
    # 加载 Part 7 训练好的 RM 检查点
    rckpt = torch.load(rm_ckpt, map_location=device)

    # 按照 RM 检查点保存的配置初始化 RewardModel 架构（包含 Transformer 编码器 + 标量输出头）。
    # 注意：RM 的默认超参（n_layer=4, n_head=4, n_embd=256）与 Policy（n_layer=2, n_head=2, n_embd=128）
    # 不同——RM 通常比 Policy 更大，因为它需要"理解"完整的文本对并做出细粒度的质量判断。
    rm = RewardModel(
        vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size=rckpt['config'].get('block_size', tok.block_size),
        n_layer=rckpt['config'].get('n_layer', 4),
        n_head=rckpt['config'].get('n_head', 4),
        n_embd=rckpt['config'].get('n_embd', 256)
    ).to(device)

    # 写入权重并切换至评估模式。
    # 注意：rm.eval() 对 Reward Model 尤其重要，因为 RM 是打分的"裁判"，
    # 如果 Dropout 随机波动，同一条 (Prompt, Response) 可能打出不同的分数，失去评估的可靠性。
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    # ─── 5. 在 Prompt 测试集中逐条采样生成与打分 ───
    # 从 Alpaca 数据集或内置 Prompt 池中抽取前 n 条 Prompt
    prompts = sample_prompts(n)
    rewards = []  # 记录每条样本在 PPO Policy 生成回答上的 RM 奖励得分

    for p in prompts:
        # ─── 5.1 格式化 Prompt 前缀 ───
        # 为什么要格式化？原始 Prompt 是纯文本（如 "Explain the purpose of attention"），
        # 但模型训练时看到的是带模板标记的结构化文本（如 "<s>[INST] Explain ... [/INST]"）。
        # 评估时必须用一模一样的模板格式包装 Prompt，否则模型会"看不懂"这个输入。
        #
        # 语法：format_prompt_only(p) 将原始问题填充进对话模板（如添加 instruction 标签）。
        #       .replace('</s>', '') 移除终止符 </s>——评估时我们不需要终止标记，
        #       因为我们希望模型"继续写下去"（生成回答），而不是在这里就判断序列结束。
        prefix = format_prompt_only(p).replace('</s>', '')

        # 将格式化后的 Prompt 转换为 token ID 列表，形状为 Python list of int
        ids = tok.encode(prefix)

        # 语法拆解：x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        #   ① ids[-tok.block_size:] → 取最后 block_size 个 Token ID（右对齐截断），
        #      确保输入不超过模型的最大上下文窗口。用 -N 切片而非前 N 个的原因：
        #      截断尾部比截断头部更合理——Prompt 的开头通常是固定的指令模板，
        #      末尾才是具体的提问内容，保留末尾信息更关键。
        #   ② [ids[-tok.block_size:]] → 外层方括号增加一个 batch 维度，
        #      形状从 (T,) 变为 (1, T)，其中 1 = batch_size（一次只生成一条）。
        #   ③ dtype=torch.long → Token ID 是整数索引（非浮点数），用 long 类型（int64）。
        #   ④ device=device → 直接放到目标设备上，避免先创建 CPU 张量再搬运。
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)

        # ─── 5.2 并行自回归生成回答 (Generation) ───
        # 语法：with torch.no_grad(): 关闭 PyTorch 的自动求导引擎（Autograd）。
        #       在推理阶段不构建计算图，历史中间激活不会被保存，大幅降低显存占用（可能节省 50%+）
        #       并加速前向传播（省去计算图构建的开销）。
        with torch.no_grad():
            # pol.generate(...)：调用 PPO 策略模型自回归生成，最多生成 128 个新 token。
            # 自回归（Autoregressive）的含义：一次生成一个 Token，每次用上一步生成的 Token
            # 作为新的输入，循环直到达到长度上限或遇到终止符。
            #
            # 参数说明：
            #   max_new_tokens=128：最多新生成 128 个 Token（约 100-200 个英文单词）
            #   temperature=0.2   ：采样温度，越低越确定。0.2 接近 greedy 但保留一定随机性，
            #                       在评估时确保生成质量稳定（不会"乱说"），同时避免完全相同的答案。
            #                       温度 T 的作用：logits/T → softmax，低温让高概率 Token 更突出。
            #   top_k=50          ：只从概率最高的 50 个 Token 中采样，过滤掉低概率的"垃圾" Token，
            #                       在保持多样性的同时防止生成无意义的文本。
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)

            # 参考模型（SFT）也生成回答（供对比参考）。
            # 两个模型使用完全相同的生成参数（same prompt, same temperature, same top_k），
            # 唯一的变量是"模型权重"——这样对比才有意义（控制变量法）。
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)

        # ─── 5.3 提取与解码生成回答 (Response Decoding) ───
        # generate() 返回的是完整的 Token 序列 = Prompt Token + 新生成的 Response Token。
        # 我们需要从中分离出"新生成的部分"，因为只有 Response 需要送入 Reward Model 打分。
        #
        # 语法：y[0].tolist() 取出第 0 个 batch 的生成结果，转换为 Python 整数列表。
        #       y 的形状为 (1, total_len)，其中 total_len = prompt_len + response_len。
        # 语法：[len(ids[-tok.block_size:]):] 使用输入 Prompt 的实际 token 长度作为起始索引切片，
        #       精确定位并截取出模型新生成的 Response token ID（即 Prompt 之后的所有 Token）。
        #       类比：一盘完整的棋谱（Prompt + 生成的下一步），用"已知棋步的长度"切出"新下的棋步"。
        resp_token_ids = y[0].tolist()[len(ids[-tok.block_size:]):]
        resp_old_token_ids = y_old[0].tolist()[len(ids[-tok.block_size:]):]

        # 语法：tok.decode(...) 将 token ID 序列重新还原为人类可读的字符串文本。
        #       这是 encode 的逆操作，底层调用分词器的 decode 方法。
        resp = tok.decode(resp_token_ids)
        resp_old = tok.decode(resp_old_token_ids)

        # ─── 5.4 格式化完整的 (Prompt, Response) 样本并利用 RM 打分 ───
        # 为什么要重新组装成完整格式？Reward Model 训练时看到的是 (Prompt, Response) 的
        # 完整拼接文本（带有对话模板标记），评估时必须使用完全相同的格式，否则 RM 无法正确"理解"
        # 这个对话对的结构，打分就会不准（格式不一致是 RLHF 评估中最常见的隐性 bug）。
        #
        # 语法：__import__('part_6.formatters', fromlist=['Example', 'format_example'])
        #       的等效写法是 from part_6.formatters import Example, format_example。
        #       这里放在循环内 import（而非文件顶部），是为了避免与 rollout.py 顶部的同名 import 冲突，
        #       同时明确表示这两个工具只在打分环节用到。
        from part_6.formatters import Example, format_example

        # 拼接成完整的对话格式文本（如 "Instruction: ...\n\nResponse: ..."），
        # 与 Reward Model 训练时的格式完全对齐（这是 RM 能正确打分的先决条件）。
        # Example 是一个数据类（dataclass/命名元组），持有 prompt 和 response 两个字符串字段。
        text = format_example(Example(p, resp))

        # 编码为 token ID 张量 → 截断至 block_size → 增加 batch 维 → 形状 (1, seq_len)
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)

        # 调用 Reward Model 前向传播计算标量奖励分值
        with torch.no_grad():
            # rm(z) 将完整的 (Prompt, Response) 文本序列送入 Transformer 编码器，
            # 取最后一个 token 位置的隐状态（即读完全文后的"整体印象"），
            # 通过线性输出头映射为一个标量分数。返回形状为 (1,) 的张量。
            #
            # 语法：[0].item() 拆解：
            #   [0]  → 从形状 (1,) 的张量中取出第 0 个元素（去掉 batch 维），得到形状 () 的标量张量
            #   .item() → 将标量张量转换为 Python float 数值，方便后续用列表收集和计算均值
            r = rm(z)[0].item()

        rewards.append(r)

    # ─── 6. 汇总计算平均奖励得分 (Average Reward Calculation) ───
    # 为什么用平均奖励而非总奖励？不同 batch 的 Prompt 数量可能不同（n 可变），
    # 平均奖励消除了样本数量的影响，使不同评估运行之间的结果具有可比性。
    #
    # 语法：sum(rewards) / max(1, len(rewards)) 计算所有测试 Prompt 的平均 Reward 标量。
    # 语法：max(1, len(rewards)) 边界防错保护——如果 rewards 列表为空（极端情况：
    #       所有 Prompt 在生成/打分环节都出了异常），分母不会被设为 0 而导致 ZeroDivisionError。
    #       此时返回 0（0/1=0），虽然不理想但不会让整个脚本崩溃。
    return sum(rewards) / max(1, len(rewards))


# ==========================================
# 命令行入口脚本 (CLI Main Entry)
# ==========================================
# 语法：if __name__ == '__main__': 是 Python 的模块入口保护机制。
#       当文件被直接运行（python eval_ppo.py）时，__name__ 的值为 '__main__'，条件成立，执行评估逻辑。
#       当文件被 import 导入（import eval_ppo）时，__name__ 的值为 'eval_ppo'，条件不成立，
#       跳过 CLI 部分，只暴露 score_policy 函数给调用者使用。
#       这种模式让同一个 .py 文件既可以作为脚本来跑，又可以作为库来导入，是 Python 开发的最佳实践。
# ==========================================
if __name__ == '__main__':
    # argparse.ArgumentParser 是 Python 标准库中的命令行参数解析器。
    # description 参数会在用户输入 --help 时显示为脚本的简介说明。
    p = argparse.ArgumentParser(description="PPO Policy Reward Score Evaluator")

    # 必填参数：PPO 策略模型 checkpoint 路径（如 runs/ppo-demo/model_last.pt）
    # required=True 表示该参数必须提供，否则解析器会报错并显示帮助信息。
    p.add_argument('--policy_ckpt', type=str, required=True, help="待评估的 PPO 策略模型检查点路径")

    # 必填参数：Part 7 训练好的 Reward Model checkpoint 路径
    p.add_argument('--reward_ckpt', type=str, required=True, help="Part 7 奖励模型检查点路径")

    # 可选参数：数据划分标记（教学微型脚本中保留该 CLI 参数以维持接口兼容性）。
    # 当前实现未使用该参数（sample_prompts 不从数据集切分），保留是为了与完整训练脚本的 CLI 风格统一。
    p.add_argument('--split', type=str, default='val[:32]', help="微型评估脚本保留的切片参数")

    # 可选参数：指定预训练 BPE 词表保存目录路径。
    # default=None 表示不指定，此时 RLHFTokenizer 会回退至 ByteTokenizer。
    p.add_argument('--bpe_dir', type=str, default=None, help="BPE 分词器目录路径")

    # parse_args() 解析命令行参数，返回包含所有参数值的命名空间对象。
    # 例如 python eval_ppo.py --policy_ckpt model.pt --reward_ckpt rm.pt
    # 则 args.policy_ckpt = 'model.pt', args.reward_ckpt = 'rm.pt'
    args = p.parse_args()

    # 执行评估逻辑并输出测试集上的平均奖励分值。
    # 将 n=16 硬编码：16 条 Prompt 能在评估速度和统计稳定性之间取得平衡。
    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.bpe_dir, n=16)

    # 语法：f"{avg_r:.4f}" 是 Python 的格式化字符串（f-string）。
    #       :.4f 表示将浮点数格式化为保留 4 位小数（如 3.1416），便于版本间精确对比。
    #       如果 avg_r=2.3，则输出 "Avg RM reward: 2.3000"。
    print(f"Avg RM reward: {avg_r:.4f}")

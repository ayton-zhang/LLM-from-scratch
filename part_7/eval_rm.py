# ==========================================
# Part 7：Reward Model 评估脚本
# 功能：加载已经训练好的奖励模型，在 chosen/rejected 偏好样本上进行推理，
#       统计模型是否能给人类偏好的回答更高分。
#
# 整体数据流：
#   偏好数据集 → 文本三元组 → token ID，并补齐到固定长度
#   → 奖励模型分别打分 chosen/rejected
#   → 比较 r_pos > r_neg → 计算 pairwise accuracy
# ==========================================

from __future__ import annotations

# argparse：解析命令行参数，例如 --ckpt 和 --split。
# torch：负责设备选择、加载 checkpoint 和执行模型推理。
import argparse, torch

# 从偏好数据集中读取 (prompt, chosen, rejected) 样本。
from data_prefs import load_preferences
# 将文本格式化、分词并补齐为奖励模型可以直接接收的 token ID 张量。
from collator_rm import PairCollator
# Part 7 的奖励模型：TransformerEncoder → masked mean pooling → 标量 reward。
from model_reward import RewardModel


def main():
    # ==========================================
    # 1. 解析评估配置
    # ==========================================
    # ArgumentParser 用来构造命令行接口；这样同一个脚本可以评估不同 checkpoint
    # 或不同数据切分，而不需要修改源代码。
    p = argparse.ArgumentParser()

    # --ckpt：必填的模型 checkpoint 路径。
    # 评估时必须知道从哪里恢复训练好的模型参数和训练配置。
    p.add_argument('--ckpt', type=str, required=True)

    # --split：要评估的数据切分，默认只取验证集前 200 条。
    # 这里沿用 Hugging Face datasets 常见的切片写法，例如 val[:200]；
    # 限制样本数可以让教学 demo 快速完成，也便于控制评估成本。
    p.add_argument('--split', type=str, default='val[:200]')

    # --cpu：布尔开关。命令行中出现该参数时值为 True，否则为 False。
    # 强制使用 CPU 适合没有 GPU 或希望复现实验环境固定的情况。
    p.add_argument('--cpu', action='store_true')

    # --bpe_dir：可选的 BPE tokenizer 目录。
    # 评估必须使用与训练奖励模型相同的 tokenizer；否则即使 vocabulary size
    # 碰巧相同，token ID 与实际含义也可能不一致，模型输入就会错位。
    p.add_argument('--bpe_dir', type=str, default=None)

    # parse_args() 读取命令行并返回 Namespace 对象，之后通过 args.ckpt 等方式取值。
    args = p.parse_args()

    # ==========================================
    # 2. 选择推理设备
    # ==========================================
    # 只有同时满足“机器有 CUDA”且“用户没有要求 CPU”时才使用 GPU。
    # Python 的 and 具有短路求值特性：如果 CUDA 不可用，后面的条件不会改变结果。
    # torch.device 是 PyTorch 统一表示 CPU/GPU 设备的对象。
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ==========================================
    # 3. 加载并整理偏好数据
    # ==========================================
    # 每个 item 通常包含：
    #   prompt：用户问题或上下文
    #   chosen：人类更偏好的回答（正样本）
    #   rejected：人类不偏好的回答（负样本）
    # 这里的 split 由命令行参数控制，例如默认读取 val[:200]。
    items = load_preferences(split=args.split)

    # 列表推导式把数据对象转换成普通三元组列表。
    # 语法：for it in items 逐个遍历样本，并按顺序取出三个字段。
    # triples 的每个元素形如 (prompt, chosen, rejected)，供 PairCollator 使用。
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    # PairCollator 负责把文本变成 token ID，并将一个 batch 内的序列补齐到
    # block_size=256。返回的 pos/neg 张量形状通常都是 (当前 batch 大小, 256)。
    # padding ID 等约定必须和训练奖励模型时保持一致，否则 mask mean pooling
    # 可能会错误地统计 padding，或者模型收到不同 tokenizer 产生的 token ID。
    col = PairCollator(block_size=256, bpe_dir=args.bpe_dir)

    # torch.load 读取 checkpoint，并用 map_location=device 将其中的张量直接映射
    # 到当前推理设备，避免 checkpoint 原本保存在 GPU、当前却只能用 CPU 时出错。
    ckpt = torch.load(args.ckpt, map_location=device)

    # checkpoint 中可能保存了训练时的模型配置；如果旧 checkpoint 没有 config，
    # 就使用空字典，后面的 cfg.get(key, default) 会自动采用默认配置。
    cfg = ckpt.get('config', {})

    # ==========================================
    # 4. 按 checkpoint 配置重建奖励模型
    # ==========================================
    # 评估不能只加载参数名称，还必须用相同的网络结构创建模型：
    # vocabulary、block size、层数、注意力头数和 embedding 宽度都要匹配。
    # cfg.get('参数名', 默认值) 表示：配置中有该参数就使用它，否则使用默认值。
    # 这能兼容没有保存完整 config 的旧 checkpoint，但最可靠的做法仍是保存并使用
    # 训练时的完整配置。
    model = RewardModel(vocab_size=cfg.get('vocab_size', col.vocab_size), block_size=cfg.get('block_size', 256),
                        n_layer=cfg.get('n_layer', 4), n_head=cfg.get('n_head', 4), n_embd=cfg.get('n_embd', 256))

    # 将 checkpoint 中名为 'model' 的参数字典写入刚创建的模型。
    # load_state_dict 会按参数名和形状逐项匹配；若结构不一致，通常会直接报错，
    # 这是在评估前尽早发现配置不匹配的重要检查。
    model.load_state_dict(ckpt['model'])

    # .to(device)：把模型参数移动到 CPU 或 GPU。
    # .eval()：切换到评估模式，关闭 Dropout 等训练期随机行为，并让 BatchNorm
    # （如果模型中存在）使用评估统计量。两个调用都返回模型本身，因此可以链式书写。
    model.to(device).eval()

    # ==========================================
    # 5. 批量推理并统计 pairwise accuracy
    # ==========================================
    # Evaluate accuracy r_pos>r_neg
    # 这里的准确率不是判断 reward 是否“绝对正确”，而是检查模型能否正确排序：
    # 对同一个 prompt，chosen 的 reward 是否高于 rejected 的 reward。
    import math

    # B 是评估 batch size，与模型 forward 中的 B（当前输入样本数）含义相同。
    # 评估时使用较小 batch 可以降低显存占用；它不会改变单条样本的预测逻辑。
    B = 16

    # correct：累计排序正确的样本对数量。
    # total：累计已经评估的样本对数量。
    correct = 0; total = 0

    # range(start, stop, step) 每次移动 B 个样本，逐批处理 triples。
    # 这样即使验证集很大，也不需要一次性把所有样本放入显存。
    for i in range(0, len(triples), B):

        # Python 切片 triples[i:i+B] 会取出当前 batch；最后一个 batch 可能少于 B 条。
        batch = triples[i:i+B]

        # collate 同时整理 chosen 和 rejected 两侧：
        #   pos：chosen token IDs，形状约为 (b, T)
        #   neg：rejected token IDs，形状约为 (b, T)
        # 其中 b 是当前 batch 大小，T 是补齐后的序列长度。
        pos, neg = col.collate(batch)

        # .to(device) 将输入张量移动到和模型相同的设备；模型与输入设备不同会报错。
        pos, neg = pos.to(device), neg.to(device)

        # ==========================================
        # 当前 batch 的前向推理
        # ==========================================
        # no_grad() 表示不记录反向传播所需的计算图。
        # 评估阶段不需要更新参数，因此这样可以节省显存并减少额外计算。
        with torch.no_grad():

            # RewardModel 将每条 token 序列映射为一个标量 reward。
            # 输入形状：(b, T)；输出形状：(b,)。
            # r_pos[k] 表示第 k 个 chosen 回答的分数。
            r_pos = model(pos)

            # 对同一批 prompt 对应的 rejected 回答打分。
            # r_neg[k] 与 r_pos[k] 必须来自同一个偏好样本对，才能进行有意义的比较。
            r_neg = model(neg)

        # 逐元素比较两个一维 reward 张量，得到形状为 (b,) 的布尔张量：
        #   True  → 模型正确判断 r_pos > r_neg
        #   False → 模型判断错误，或两者分数相等（相等也不满足严格大于）
        # .sum() 统计 True 的数量，.item() 把只有一个元素的张量转换成 Python 数字，
        # 方便累加到普通整数 correct 中。
        correct += (r_pos > r_neg).sum().item()

        # pos.size(0) 读取输入张量第 0 维，也就是当前 batch 中的样本对数量 b。
        # 使用实际 batch 大小而不是固定的 B，可以正确处理最后一个不满 batch 的批次。
        total += pos.size(0)

    # ==========================================
    # 6. 汇总并输出评估结果
    # ==========================================
    # pairwise accuracy = 排序正确的样本对数 / 总样本对数。
    # max(1, total) 防止数据集为空时发生除以 0；此时输出的 accuracy 会是 0.0。
    acc = correct / max(1, total)

    # f-string 将变量嵌入字符串；.3f 表示 accuracy 保留三位小数。
    # pairs 是评估的偏好对数量，accuracy 越高表示模型越能复现数据中的偏好排序。
    print(f"pairs={total}  accuracy (r_pos>r_neg) = {acc:.3f}")


# Python 文件入口保护：只有直接运行 python eval_rm.py 时才执行 main()；
# 如果该文件被其他模块 import，则不会自动开始加载数据和评估。
if __name__ == '__main__':
    main()

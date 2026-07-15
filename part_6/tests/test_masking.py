# ==========================================
# 单元测试：SFTCollator 的 Label Masking
# ==========================================
# 测试 collator_sft.py 中最关键的功能——prompt 区域的 label 屏蔽。
#
# 为什么这是 SFT 中最需要测试的功能？
#   如果 label masking 有 bug（如把 response 也屏蔽了，或漏掉了 prompt），
#   训练过程不会报错——loss 照常下降，但模型实际上学错了东西：
#     - 屏蔽漏了 prompt：模型浪费算力学习"背诵用户问题"
#     - 误屏蔽了 response：模型完全不学习回答，loss 永远降不下去
#   这两种 bug 在训练中都是"静默"的，只有通过单元测试才能提前发现。

from collator_sft import SFTCollator
from formatters import Example


# ==========================================
# test_masking_sets_prompt_to_ignore()
# ==========================================
# 核心验证：经过 collator.collate() 处理后的 label 张量中，
# 必须存在值为 -100 的位置（即 prompt 区域被屏蔽）。
#
# 测试策略：
#   这是一种"冒烟测试"（smoke test）而不是穷举测试——
#   不验证每个具体位置的值，只验证"至少有一些 -100 存在"。
#   这种测试更稳健：即使 tokenizer 的具体编码结果变化，测试也不会误报。
def test_masking_sets_prompt_to_ignore():
    # ─── 创建 collator 实例 ───
    # block_size=256：足够大的上下文，避免截断干扰测试结果
    # bpe_dir 指向 Part 4 训练的 tokenizer（测试依赖预训练产物）
    col = SFTCollator(block_size=256, bpe_dir='../part_4/runs/part4-demo/tokenizer')

    # ─── 准备测试数据 ───
    # 一条极简的 (prompt, response) 对
    text = "This is a tiny test."

    # ─── 执行整理操作 ───
    # col.collate([(text, "OK")]) 将单条样本处理为 (x, y) 张量对。
    # 注意：batch 参数是 List[Tuple[str,str]]，外层列表表示 batch_size=1。
    # 返回值：
    #   x：输入 token ID 张量，形状 (1, 256)
    #   y：目标 label 张量，形状 (1, 256)，prompt 区域应为 -100
    x, y = col.collate([(text, "OK")])

    # ─── 断言：必须存在被屏蔽的位置 ───
    # All labels up to response marker should be -100
    # 语法：(y[0] == -100) 逐元素比较，返回形状 (256,) 的布尔张量。
    # .sum() 统计 True（即值为 -100 的位置）的个数。
    # 断言 > 0：必须至少有一个位置被屏蔽。
    #
    # 为什么不用更精确的断言（如比较具体哪些位置是 -100）？
    #   因为不同 tokenizer 对同一文本的编码结果不同（BPE vs Byte），
    #   prompt 的 token 数量和边界位置会变化。用"至少有一个"的宽松断言，
    #   让测试在不同 tokenizer 下都能通过，同时确保核心功能存在。
    #
    # 如果 masking 逻辑彻底失效（比如注释掉了 for i in range(n_prompt-1): y[i]=-100），
    # 这个断言会失败——没有 -100 存在，测试立即报告问题。
    assert (y[0] == -100).sum() > 0

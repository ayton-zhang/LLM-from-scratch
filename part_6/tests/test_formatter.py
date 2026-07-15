# ==========================================
# 单元测试：format_example 和 format_prompt_only
# ==========================================
# 测试 formatters.py 中的两个格式化函数是否正常工作。
#
# 为什么需要这个测试？
#   SFT 训练和推理强烈依赖对话模板的一致性——如果格式出错，
#   模型会"懵掉"（训练时见的格式和推理时不同）。这个测试是
#   一道"安全闸"，确保模板修改后不会破坏已有功能。

from formatters import Example, format_example, format_prompt_only


# ==========================================
# test_template_contains_markers()
# ==========================================
# 验证点：
#   1. format_example 的输出必须包含 ### Instruction: 和 ### Response: 标记
#      ——这两个标记是模型识别指令/回答边界的关键信号
#   2. format_prompt_only 的输出必须以 ### Response:\n 结尾
#      ——推理时需要这个"半截"格式来触发模型生成回答
def test_template_contains_markers():
    # ─── 测试 format_example ───
    # 创建一个最简单的 Example 对象
    ex = Example("Say hi", "Hello!")

    # 调用 format_example 获取完整格式化文本
    s = format_example(ex)

    # 语法：assert 条件，条件为 False 时抛出 AssertionError，pytest 会捕获并报告。
    # `A in s and B in s` 检查两个子字符串是否都存在于 s 中。
    # 这是测试的核心断言：模板标记是否存在。
    assert "### Instruction:" in s and "### Response:" in s

    # ─── 测试 format_prompt_only ───
    # format_prompt_only 只生成 prompt 部分，response 留空
    p = format_prompt_only("Explain transformers.")

    # 语法：str.endswith(suffix) 检查字符串是否以 suffix 结尾。
    # 这里用 or 连接两种可能：
    #   "### Response:\n"      → collator 中去掉 </s> 后的格式
    #   "### Response:\n</s>"  → 原始模板格式（含结束标记）
    # 两种都可以接受，体现了测试的"宽容性"——不因实现细节变化而误报。
    assert p.endswith("### Response:\n") or p.endswith("### Response:\n</s>")

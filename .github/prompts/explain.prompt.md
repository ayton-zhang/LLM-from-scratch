---
mode: agent
description: 用中文解释代码并保存为学习文档
---

请完成以下任务：

1. 用中文解释选中的代码，包含两个部分：
- 语法用法：逐一说明关键语法、API 或 PyTorch 操作（如 `.view()`、`nn.ModuleList`、`F.scaled_dot_product_attention`）的作用。
- 代码逻辑：按执行顺序梳理流程，解释每一步在做什么、为什么这么做。

2. 将解释保存为 Markdown 学习文档，而不是只输出在对话框。
- 文档目录：`notes/explanations/`
- 文件名：`<源文件名>-<代码主题>-YYYYMMDD-HHMMSS.md`
- 命名规则：
	- `<源文件名>`：去扩展名（例如 `attn_modern.py` -> `attn_modern`）
	- `<代码主题>`：优先使用类名/函数名（如 `CausalSelfAttentionModern_init`）；若无法确定，则使用 `selection_notes`
	- 仅保留小写字母、数字、下划线；空格与特殊符号统一替换为下划线
- 若目录不存在，请先创建。

3. 文档结构要求：
- 标题：`[<源文件名>] <代码主题> 代码解释笔记`
- 小节：原始代码、语法用法、代码逻辑、关键知识点总结
- 使用中文，术语可保留英文原名（如 `tensor`、`batch size`）
- 面向初学者，解释不要过于简略
- 如果涉及本项目模块（如 `GPTModern`、`RollingKV`），结合项目上下文说明

4. 在对话中只简要返回：
- 已保存文件路径
- 本次笔记的 3 条关键收获

待解释的代码：
${selection}

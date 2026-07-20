---
module: exercises
path: StudyVault/12-Exercises
keywords: practice, sft, masking, part-6
---

# Part 6 — 练习

#practice #onboarding #module-part-6

## Related Modules
- [[Part 6 — Supervised Fine-Tuning]]

## 练习 1 — Code Reading [trace]
> `SFTCollator` 如何从一个 example 得到 response-only loss labels？
> [!answer]- 查看答案
> 格式化 prompt+response、tokenize、构造 shifted labels；prompt 对应 labels 设为 `-100`，response 仍接受监督。

## 练习 2 — Recall [recall]
> 为什么 mask 到 `n_prompt - 1` 而不是 `n_prompt`？
> [!answer]- 查看答案
> shift 后该边界 label 预测的是首个 response token；多 mask 一个会丢失这项监督。

## 练习 3 — Configuration [config]
> 改 padding ID 时还必须检查哪些组件？
> [!answer]- 查看答案
> collator 的 padding/label mask、reward model pooling mask，以及 rollout batching 都依赖同一约定。

## 练习 4 — Debugging [debug]
> `orchestrator.py` 没有跑预期的 masking 测试，怎么办？
> [!answer]- 查看答案
> 该文件中相应 `run` 调用被注释；进入 `part_6` 后直接执行 `python -m pytest -q`。

## 练习 5 — Extension [extend]
> 若新增 chat template，最小验证方案？
> [!answer]- 查看答案
> 修改 `formatters.py`，为边界 token/标签写精确断言，再重跑 Part 6、7、8、9 的相关测试或下游 smoke test。

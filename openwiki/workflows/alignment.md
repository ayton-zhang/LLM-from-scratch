---
type: 工作流指南
title: 对齐工作流
description: 跟踪 Parts 6–9 从仅监督回答的 SFT、成对 reward modeling 到简化 PPO 和 GRPO 训练循环的 checkpoint 依赖与目标规则。
resource: part_6/train_sft.py
tags: [对齐, sft, reward-model, ppo, grpo]
---

# 对齐工作流

Parts 6–9 构成分阶段的教学管线。它们复用[模型与训练架构](../architecture/model-and-training.md)产生的模型和 tokenizer，并在各阶段产生下一阶段所需工件。命令与具体工件位置由[测试与运行](../operations/testing-and-runs.md)维护。

```text
Part 4 pretrained GPTModern + BPE tokenizer
                 |
                 v
Part 6 SFT checkpoint ------------------------+
                 |                            |
                 v                            v
Part 7 reward-model checkpoint        frozen reference policy
                 |                            |
                 +------------+---------------+
                              v
                    Part 8 PPO 或 Part 9 GRPO
```

Part 5 MoE 不接入这条链。

## Part 6：supervised fine-tuning

`part_6/dataset_sft.py` 加载 Alpaca-style instruction rows，并提供极小 fallback，暴露 instruction/input/output 字段。`formatters.py` 将它们转换为稳定的 prompt/response template。`collator_sft.py` 随后：

1. token 化 prompt 和 response；
2. 构造 causal sequence 与右移一位的 next-token labels；
3. 用 `-100` 屏蔽 prompt supervision；
4. 用 token ID `2` padding input IDs，并用 `-100` padding labels。

masking boundary 有意保留“预测第一个 response token”的 label：右移之后，它只 mask 到 `n_prompt - 1`，而非整个 prompt-length prefix。这一历史修正是 response-only training 的核心。

`train_sft.py` 复用 `GPTModern`，可从 Part 4 pretrained checkpoint 初始化。demo 写入 `part_6/runs/sft-demo/model_last.pt`；它既是 trainable starting policy，也是 PPO/GRPO 的 frozen reference。

当前 `part_6/orchestrator.py` 默认会运行 demo，`--no-demo` 才跳过；其中 `test_formatter.py` 和 `test_masking.py` 的编排调用已被注释。因此，若需要实际单元测试，应从 `part_6/` 显式运行 `python -m pytest -q`，不要把 `python orchestrator.py` 视为测试入口。

## Part 7：成对 reward model

`part_7/data_prefs.py` 加载 chosen/rejected preference pairs（或两个 fallback examples）。`collator_rm.py` 使用 SFT template 格式化并 token 化每一侧。与 causal LM 不同，`model_reward.py` 使用 bidirectional `TransformerEncoder`、对 non-padding tokens 做 masked mean pooling，并接 scalar projection。

`loss_reward.py` 提供两种 loss：

- Bradley–Terry：`mean(softplus(-(r_chosen - r_rejected)))`。
- Margin ranking：要求 `r_chosen >= r_rejected + margin`，默认 margin 为 `1.0`。

`eval_rm.py` 将 `r_chosen > r_rejected` 的比例报告为 pairwise accuracy。demo 的 `part_7/runs/rm-demo/model_last.pt` 为两条 RL 工作流提供 scalar rewards。

## Part 8：简化 PPO RLHF

`part_8/policy.py` 用 toy value head 包装 SFT LM。`train_ppo.py` 将 SFT model 克隆成 trainable policy 和 frozen reference：每个 prompt 生成一个 completion，用 reward model 对格式化文本评分，再借助 `rollout.py` 中的工具选择 response-token log probabilities 与 values。

更新规则经过刻意压缩：

- probability ratio：`exp(new_logp - old_logp)`；
- clipped PPO policy surrogate；
- value loss：普通 MSE；
- sampled-token “entropy”：选中 token 的负 log probability 均值；
- 默认 total loss：`policy + 0.5 * value - ent_coef * entropy`。

reference divergence 会在计算 advantages 前作为 token reward shaping 加入。scalar reward 只放在最后一个 selected position；其他 response positions 只有零奖励减 KL cost。returns 等于这些 immediate shaped rewards，advantages 为 `returns - old_values` 再归一化。

尽管暴露 `gamma` 和 `lambda` 参数，该循环没有实现 discounted returns 或 GAE；每次 fresh rollout 只做一轮 update。因此这是讲解 PPO mechanics 的实现，不是完整 RLHF trainer。

## Part 9：简化 GRPO

`part_9/train_grpo.py` 为每个 prompt 生成 `group_size` 个 completions，用同一 reward model 评分，减去每 prompt 的 group mean，并将 trajectory advantage 广播到所有 response tokens。`grpo_loss.py` 组合 PPO-style clipped policy objective 与基于 `mean(new_logp - ref_logp)` 的显式 reference term。

与 PPO 不同，GRPO 没有 value loss。但 Part 9 仍实例化复制的 `PolicyWithValue`；其 value head 被忽略。advantages 在 token broadcasting 后归一化，因此较长 response 会向展平 loss 提供更多条目。实现把 reference-difference term 称为 “KL”，但它是 sampled log-probability difference，而不是完整分布 KL。

## 跨阶段不变量与风险

- **Tokenizer identity：**Parts 6–9 必须使用与 pretrained/SFT policy 同一份保存的 BPE tokenizer；仅 vocabulary-size 相等并不够。
- **模型维度：**layer count、head count、embedding width、block size 和 vocabulary 必须匹配 checkpoint；demos 使用与 Part 4 smoke run 对应的小型 `2/2/128` 模型。
- **Padding：**alignment collators 和 rollout batching 假定 padding ID 为 `2`；改动后须检查 masks 和 reward pooling。
- **序列边界：**空或截断的 responses 会破坏 terminal-reward indexing 与 masking 假设。
- **Reward-model contract：**RL 加载的 RM architecture/config 必须匹配其 checkpoint 与 tokenizer。
- **评估边界：**`eval_ppo.py` 是小型比较脚本，复制到 Part 9，含未使用或 copy-forward options；不是 benchmark harness。

## 变更指引

改动 formatting 或 tokenization 时，在 `part_6/tests/test_masking.py` 增加精确 boundary assertions，并运行 reward/RL tests，因为每个后续阶段都 import 或重现这些约定。改动 objective 时，应增加 clipping、masks、KL sign、group baselines 和 gradient flow 的数值测试。当前没有端到端兼容性覆盖，所以任何 checkpoint schema 改动都应手动 smoke-test 到下一阶段。
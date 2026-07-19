---
type: Workflow Guide
title: Alignment Workflows
description: Traces Parts 6–9 from response-only supervised fine-tuning through pairwise reward modeling to the repository's simplified PPO and GRPO training loops, including checkpoint dependencies and objective rules.
tags: [alignment, sft, reward-model, ppo, grpo]
resource: part_6/train_sft.py
---

# Alignment workflows

Parts 6–9 form a staged educational pipeline. They reuse the model and tokenizer produced by [Model and training architecture](../architecture/model-and-training.md), and each stage emits artifacts required by the next. Commands and concrete artifact locations live in [Testing and runs](../operations/testing-and-runs.md).

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
                    Part 8 PPO or Part 9 GRPO
```

Part 5 MoE is not integrated into this chain.

## Part 6: supervised fine-tuning

`part_6/dataset_sft.py` loads Alpaca-style instruction rows with a tiny fallback and exposes instruction/input/output fields. `formatters.py` converts them to a stable prompt and response template. `collator_sft.py` then:

1. tokenizes prompt and response;
2. builds a causal sequence and next-token-shifted labels;
3. masks prompt supervision with `-100`;
4. pads input IDs with token ID `2` and labels with `-100`.

The masking boundary intentionally leaves the label that predicts the first response token visible: it masks through `n_prompt - 1`, not the entire prompt-length prefix after shifting. This behavior came from a historical correction and is central to response-only training.

`train_sft.py` reuses `GPTModern` and can initialize from Part 4's pretrained checkpoint. The demo writes `part_6/runs/sft-demo/model_last.pt`; this becomes both the trainable starting policy and frozen reference for PPO/GRPO.

## Part 7: pairwise reward model

`part_7/data_prefs.py` loads chosen/rejected preference pairs (or two fallback examples). `collator_rm.py` formats and tokenizes each side using the SFT template. Unlike the causal LM, `model_reward.py` uses a bidirectional `TransformerEncoder`, masked mean pooling over non-padding tokens, and a scalar projection.

Two losses are available in `loss_reward.py`:

- Bradley–Terry: `mean(softplus(-(r_chosen - r_rejected)))`.
- Margin ranking: enforce `r_chosen >= r_rejected + margin`, with default margin `1.0`.

`eval_rm.py` reports pairwise accuracy as the fraction where `r_chosen > r_rejected`. The demo's reward checkpoint at `part_7/runs/rm-demo/model_last.pt` supplies scalar rewards to both RL workflows.

## Part 8: simplified PPO RLHF

`part_8/policy.py` wraps the SFT LM with a toy value head. `train_ppo.py` clones the SFT model into a trainable policy and frozen reference, generates one completion per prompt, scores formatted text with the reward model, and selects response-token log probabilities and values through utilities in `rollout.py`.

The update rules are intentionally compact:

- probability ratio: `exp(new_logp - old_logp)`;
- clipped PPO policy surrogate;
- value loss: plain MSE;
- sampled-token “entropy”: mean negative selected-token log probability;
- total loss defaults to `policy + 0.5 * value - ent_coef * entropy`.

Reference divergence is applied as token reward shaping before advantages are computed. The scalar reward appears only at the terminal selected position; other response positions receive zero reward minus KL cost. Returns equal these immediate shaped rewards, and advantages are `returns - old_values` followed by normalization.

Despite exposed `gamma` and `lambda` arguments, this loop does not implement discounted returns or GAE. It performs one update pass per fresh rollout. This is a teaching implementation of PPO mechanics, not a full RLHF trainer.

## Part 9: simplified GRPO

`part_9/train_grpo.py` generates `group_size` completions per prompt, scores each with the same reward model, subtracts the per-prompt group mean, and broadcasts that trajectory advantage to all response tokens. `grpo_loss.py` applies a PPO-style clipped policy objective plus an explicit reference term based on `mean(new_logp - ref_logp)`.

Unlike PPO, GRPO has no value loss. However, Part 9 still instantiates the copied `PolicyWithValue`; its value head is simply ignored. Advantages are normalized after token broadcasting, so longer responses contribute more entries to the flattened loss. The implementation also calls the reference-difference term “KL,” though it is a sampled log-probability difference rather than a full-distribution KL calculation.

## Cross-stage invariants and risks

- **Tokenizer identity:** Parts 6–9 must use the same saved BPE tokenizer as the pretrained/SFT policy. Vocabulary-size equality is not enough.
- **Model dimensions:** layer count, head count, embedding width, block size, and vocabulary must match the checkpoint; demos use a small `2/2/128` model tied to the Part 4 smoke run.
- **Padding:** alignment collators and rollout batching assume padding ID `2`. Changing it requires checking masks and reward pooling.
- **Sequence boundaries:** empty or truncated responses can invalidate terminal-reward indexing and masking assumptions.
- **Reward-model contract:** the RM architecture/config loaded during RL must match its checkpoint and tokenizer.
- **Evaluation limits:** `eval_ppo.py` is a small comparison script, duplicated into Part 9, and contains unused/copy-forward options; it is not a benchmark harness.

## Change guidance

When changing formatting or tokenization, add exact boundary assertions to `part_6/tests/test_masking.py` and run reward/RL tests because every later stage imports or reproduces those conventions. Objective changes should add numerical tests for clipping, masks, KL sign, group baselines, and gradient flow. End-to-end compatibility is currently not covered, so any checkpoint schema change should be smoke-tested through the next stage manually.

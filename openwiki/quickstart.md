---
type: Quickstart
title: LLM from Scratch Repository Quickstart
description: Entry point for a nine-part PyTorch curriculum that builds a Transformer language model, scales its training stack, and introduces MoE, supervised fine-tuning, reward modeling, PPO, and GRPO.
tags: [pytorch, llm, curriculum, quickstart]
resource: README.md
---

# LLM from Scratch

This repository is an executable, learning-oriented PyTorch curriculum. Its nine directories progress from attention primitives to a tiny GPT, modern inference features, scalable pretraining, Mixture-of-Experts, and an educational alignment stack. It is not a packaged library or production service: each part has local imports, scripts, tests, and usually an `orchestrator.py` intended to run with that part as the working directory.

## Start here

Create the Python 3.11 environment described by `README.md`:

```bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

Then run a focused part from its own directory. For example:

```bash
cd part_1
python orchestrator.py --visualize
```

Or run only that part's tests:

```bash
cd part_6
python -m pytest -q
```

Working directory matters because modules use sibling imports such as `from gating import TopKGate`; root-level `pytest` collection is therefore not the supported path. See [Testing and runs](operations/testing-and-runs.md) for all commands, Docker setup, artifacts, and prerequisites.

## Curriculum map

| Part | Purpose | Main entrypoint | Output or dependency |
|---|---|---|---|
| 1 | Positional encoding, causal attention, MHA, FFN, residual Transformer block | `part_1/orchestrator.py` | Shape/math demonstrations and optional attention images |
| 2 | Byte-tokenized tiny GPT, training, sampling, validation | `part_2/orchestrator.py` | `part_2/runs/min-gpt/model_best.pt` |
| 3 | RMSNorm, RoPE, SwiGLU, GQA, KV cache, sliding-window attention | `part_3/orchestrator.py` | Modern model implementation used downstream |
| 4 | BPE, AMP, gradient accumulation, LR scheduling, checkpoints, logging | `part_4/orchestrator.py` | BPE tokenizer and pretrained checkpoint |
| 5 | Top-k MoE routing and dense/MoE hybrid FFNs | `part_5/orchestrator.py` | Standalone component demo; not wired into Parts 6–9 |
| 6 | Response-only supervised fine-tuning | `part_6/orchestrator.py --demo` | SFT policy checkpoint |
| 7 | Pairwise reward modeling | `part_7/orchestrator.py --demo` | Reward-model checkpoint |
| 8 | Educational on-policy PPO RLHF | `part_8/orchestrator.py --demo` | PPO policy checkpoint |
| 9 | Educational group-relative policy optimization | `part_9/orchestrator.py --demo` | GRPO policy checkpoint |

The first five parts are explained in [Model and training architecture](architecture/model-and-training.md). Parts 6–9 form the checkpoint and data flow described in [Alignment workflows](workflows/alignment.md). Those two domains meet at the Part 3 `GPTModern` implementation and the Part 4 tokenizer/pretrained checkpoint; [Testing and runs](operations/testing-and-runs.md) explains how those assets are created and kept compatible.

## How to read the repository

1. Begin with each part's `orchestrator.py`; it is the most reliable map of supported tests and demos.
2. Follow the data path before the model path: tokenizer/dataset or collator, model, loss, training loop, checkpoint, sampler/evaluator.
3. Treat the extensive inline Chinese annotations and `notes/explanations/` as learning aids, but verify behavior against executable code. Some comments were copied forward or predate behavioral fixes.
4. Preserve cross-stage invariants when experimenting: model dimensions, vocabulary size, tokenizer files, padding conventions, and checkpoint paths must agree.

Recent history confirms the repository's educational progression: Parts 6–9 were added as sequential alignment stages, while later commits concentrated on correcting SFT label shifting/masking, PPO rollouts, RollingKV behavior, tests, and detailed annotations. Documentation therefore distinguishes intended concepts from known simplified or disconnected implementations.

## Backlog

- **Generated model quality and benchmark results** — anchors: `part_*/runs/`; deferred because run artifacts and reproducible metrics are not committed.
- **CI automation** — anchor: `.github/workflows/`; deferred because the current workflow directory is untracked and is not established repository behavior.

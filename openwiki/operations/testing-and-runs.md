---
type: Runbook
title: Testing and Runs
description: Practical runbook for setting up the Python or Docker environment, running each curriculum part, locating generated checkpoints, and understanding test coverage and operational caveats.
tags: [testing, operations, checkpoints, docker]
resource: requirements.txt
---

# Testing and runs

This runbook executes and verifies the concepts in [Model and training architecture](../architecture/model-and-training.md) and the staged [Alignment workflows](../workflows/alignment.md). The repository has no root package configuration; local working directories are part of its runtime contract.

## Environment

Use Python 3.11 and `pip install -r requirements.txt`. Important pinned packages include PyTorch 2.8, pytest, NumPy, Hugging Face `tokenizers`/`datasets`, and TensorBoard. CUDA is optional for small tests, but training and generation are substantially more practical on a GPU. WandB logging requires a separate `wandb` installation because it is not in `requirements.txt`.

Docker provides the alternative development environment:

```bash
docker compose build
docker compose up -d
docker compose exec llm-lab bash
```

`Dockerfile` builds a Python 3.11 image at `/app`; `docker-compose.yml` bind-mounts the repository there, requests all NVIDIA GPUs, and sleeps indefinitely for interactive use. Direct `docker run` defaults to `python part_1/demo_mha_shapes.py`, while Compose overrides that command.

## Per-part checks

Run commands from the indicated directory.

| Part | Tests/checks | Demo behavior |
|---|---|---|
| 1 | `python orchestrator.py` | Add `--visualize` to write attention PNGs under `part_1/out/` |
| 2 | `python orchestrator.py` | Always trains, samples, and evaluates; enables AMP and `torch.compile` |
| 3 | `python orchestrator.py --skip-demo` | Omit `--skip-demo` to also generate 200 tokens with a window/sink configuration |
| 4 | `python orchestrator.py --no-demo` | Omit `--no-demo` to train a small BPE model and sample from it |
| 5 | `python orchestrator.py --no-demo` | Demo runs by default unless disabled |
| 6 | `python orchestrator.py` | Add `--demo`; requires Part 4 artifacts |
| 7 | `python orchestrator.py` | Add `--demo`; requires Part 4 tokenizer |
| 8 | `python orchestrator.py` | Add `--demo`; requires Parts 4, 6, and 7 artifacts |
| 9 | `python orchestrator.py` | Add `--demo`; requires Parts 4, 6, and 7 artifacts |

For a single part's entire test directory:

```bash
cd part_5
python -m pytest -q
```

Orchestrators run selected test files in fail-fast subprocesses with the correct `cwd`. Avoid `pytest` from repository root: tests import sibling modules as top-level names and collection will fail without import-path manipulation.

## Artifact chain

The default demonstrations expect these paths:

```text
part_2/runs/min-gpt/model_best.pt
part_4/runs/part4-demo/
  model_last.pt
  tokenizer/
part_6/runs/sft-demo/model_last.pt
part_7/runs/rm-demo/model_last.pt
part_8/runs/ppo-demo/model_last.pt
part_9/runs/grpo-demo/model_last.pt
```

To run the full alignment chain, execute the Part 4 demo first, then Parts 6 and 7 demos, then either Part 8 or Part 9. Part 6's model dimensions are explicitly matched to the Part 4 smoke model (`n_layer=2`, `n_head=2`, `n_embd=128`). Do not substitute an arbitrary checkpoint without matching CLI architecture arguments and tokenizer.

Generated `runs/`, logs, tokenizers, and checkpoints are operational artifacts, not guaranteed repository contents. Part 4 stores a tokenizer path alongside checkpoints; moving the run tree can invalidate that path.

## What the tests establish

- **Part 1:** attention/mask shapes and finite outputs.
- **Part 2:** byte-tokenizer behavior and exact next-token dataset shift.
- **Part 3:** RMSNorm, RoPE shape/value behavior, and bounded standalone `RollingKV` shape.
- **Part 4:** BPE save/load lifecycle, scheduler bounds/progression, and checkpoint shape smoke behavior.
- **Part 5:** gate shapes, MoE integration, hybrid block output, and minimal gradient flow.
- **Part 6:** formatting and basic existence of masked labels.
- **Part 7:** reward forward shape and Bradley–Terry monotonicity.
- **Part 8:** policy forward shape and scalar PPO loss.
- **Part 9:** scalar/finite GRPO loss and returned diagnostics.

These are component tests, not model-quality tests. They do not establish checkpoint compatibility across parts, tokenizer identity, cached-vs-uncached equivalence, exact SFT boundary masking, complete router/expert gradients, PPO rollout correctness, or GRPO group-baseline construction.

## Troubleshooting and guardrails

- **Import errors:** confirm the shell is inside `part_N/`; do not “fix” imports globally without considering every orchestrator.
- **Missing alignment checkpoint:** run the upstream demo named in the artifact chain.
- **Checkpoint size mismatch:** use the architecture and vocabulary that created it. Prefer self-describing checkpoints such as Part 2's `model_best.pt` and Part 4's full checkpoint format.
- **Tokenizer load failure:** verify the original saved tokenizer directory and `tokenizer.json`; avoid silently creating an untrained BPE tokenizer as a fallback.
- **CPU slowness or compile issues:** Part 2's orchestrator turns on AMP and compile; invoke `train.py` directly with conservative flags when diagnosing CPU environments.
- **Long-generation memory:** do not assume Part 3's tested `RollingKV` bounds the cache used by `GPTModern.generate()`; inspect cache length explicitly.
- **Scheduler surprises:** in Part 4, scheduler horizon and `--steps` stopping logic are calculated differently; test non-default epoch/dataset combinations.

## Before changing a stage

1. Run that part's tests from its directory.
2. Identify upstream inputs and downstream checkpoint consumers.
3. Make the change in source code, not generated run artifacts.
4. Re-run component tests and the smallest relevant orchestrator demo.
5. For tokenizer, checkpoint, formatting, padding, or model-config changes, manually smoke-test the immediately downstream stage because current tests do not cover the boundary.

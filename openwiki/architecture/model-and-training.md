---
type: Architecture Guide
title: Model and Training Architecture
description: Explains how Parts 1–5 evolve attention primitives into GPTModern, BPE-based training infrastructure, cached inference, and a standalone sparse Mixture-of-Experts feed-forward layer.
tags: [architecture, transformer, pretraining, moe]
resource: part_3/model_modern.py
---
# Model and training architecture

Parts 1–4 form a deliberate progression from transparent tensor math to a reusable modern language model and training stack. Part 5 explores MoE as a separate feed-forward replacement. The resulting `GPTModern` model and Part 4 assets are then reused by the [alignment workflows](../workflows/alignment.md), while their executable checks and artifact paths are catalogued in [Testing and runs](../operations/testing-and-runs.md).

## Part 1: make attention observable

Part 1 favors explicit operations over optimization. `part_1/multi_head.py` projects `(B,T,C)` into Q/K/V, reshapes them to `(B,H,T,D)`, applies scaled causal attention, merges heads, and returns both output and attention weights. `part_1/block.py` wraps attention and a 4× GELU FFN in pre-normalized residual connections.

The NumPy demo, shape walkthrough, and optional heatmaps exist to make the matrix operations inspectable before training is introduced. This code is educational and standalone; later models reimplement the architecture rather than importing Part 1.

## Part 2: train the first causal LM

The first end-to-end flow is:

```text
UTF-8 file
  -> byte IDs (0..255)
  -> 90/10 train/validation split
  -> random x windows and one-token-shifted y windows
  -> token + learned position embeddings
  -> pre-norm causal Transformer blocks
  -> LM head and flattened cross-entropy
  -> validation, checkpoint, autoregressive sampling
```

The data shift is implemented in `part_2/dataset.py`; architecture and generation are in `part_2/model_gpt.py`; `part_2/train.py` owns AdamW, clipping, optional AMP/compile, periodic evaluation, and checkpoints. Generation crops context to `block_size` and recomputes the full window on every token, which motivates Part 3's cache.

`model_best.pt` is the safest sampling artifact because it carries the model configuration used by `part_2/sample.py`. The final checkpoint format is less self-describing and may not reload a non-default architecture correctly.

## Part 3: modern model and cached inference

`part_3/model_modern.py` removes learned positional embeddings and composes feature-selectable modern components:

- `rmsnorm.py`: scale normalization without mean centering.
- `rope_custom.py`: rotary position information applied to attention queries and keys.
- `swiglu.py`: gated FFN activation.
- `attn_modern.py`: causal attention with optional grouped-query attention, sliding windows, attention sinks, and K/V input/output.
- `kv_cache.py`: cache containers plus a bounded `RollingKV` that retains the first `sink` and latest `window` positions.

Autoregressive generation performs one full prompt **prefill**, then passes one new token at each **decode** step while reusing prior K/V. This changes repeated attention setup from full-window recomputation to incremental reuse.

### Cache caveat

The separately tested `RollingKV` enforces `length <= sink + window`, but `GPTModern.generate()` does not wire that object into its main cache path. `attn_modern.py` crops K/V for an attention computation, then creates the returned cache from the prior uncropped cache plus new K/V. Long-running generation can therefore keep growing the returned cache. In addition, using cache length as RoPE's next position is insufficient once old middle tokens are discarded; a production streaming design needs explicit absolute-position tracking. Treat the sliding-window demo as instructional, not a production memory guarantee.

## Part 4: scale the training loop

`part_4/train.py` imports `GPTModern` from Part 3 and fixes RMSNorm, RoPE, and SwiGLU on for training. Its pipeline is:

```text
text file -> train/load BPE -> tokenize corpus -> overlapping shifted windows
-> DataLoader -> GPTModern -> cross-entropy
-> AMP-scaled and/or accumulated gradients -> AdamW
-> warmup + cosine scheduler -> logger -> checkpoints
```

The surrounding modules separate concerns:

- `tokenizer_bpe.py` trains and persists a Hugging Face BPE tokenizer.
- `dataset_bpe.py` creates shifted training windows; despite “streaming” language, it reads and tokenizes the full file in memory.
- `amp_accum.py` handles autocast/scaling and gradient accumulation.
- `lr_scheduler.py` implements warmup plus cosine decay.
- `checkpointing.py` saves model, optimizer, scheduler, scaler, step, and model configuration.
- `logger.py` supports TensorBoard and optional WandB; WandB is dynamically imported but not pinned in `requirements.txt`.

Resume and downstream use depend on the original tokenizer path and compatible model/vocabulary dimensions. Moving a run directory can break its recorded tokenizer location. The training stop condition also follows `--steps`, while scheduler horizon derives from dataset/epoch calculations; unusual combinations can make those horizons diverge.

## Part 5: sparse experts as an FFN alternative

Part 5 is a standalone component study, not part of the Part 4–9 checkpoint chain. `part_5/gating.py` computes softmax expert probabilities and selects top-k experts for every token. `part_5/moe.py` dispatches tokens through independent MLPs with explicit Python loops and combines outputs using selected gate weights.

Important semantics:

- Selected top-k weights are not renormalized, so their sum can be below one.
- The balancing term is `E * sum(importance * load)`, with soft-probability importance and top-1 assignment frequency.
- There is no capacity factor, dropped-token policy, expert parallelism, or all-to-all communication.
- `HybridFFN` in `block_hybrid.py` returns `alpha * Dense(x) + (1-alpha) * MoE(x)`, defaulting to `alpha=0.5`; both branches execute even at an endpoint value.

This implementation makes routing mechanics easy to inspect but should not be read as a performance-oriented MoE system. `part_5/README.md` explicitly frames production expert parallelism as outside scope.

## Change guidance

- Attention changes should preserve `(B,H,T,D)` contracts and causal masking; run Part 1 and Part 3 tests.
- Model configuration changes must propagate through checkpoint creation, loading, sampling, and every downstream stage.
- Tokenizer changes affect Parts 4 and 6–9 together. Never compare or continue checkpoints with mismatched token IDs merely because vocabulary sizes match.
- Cache work needs stronger tests than currently exist: compare cached and uncached logits/generation and exercise sequences longer than the configured window.
- MoE changes should test routing gradients and per-expert utilization, not only output shapes and “some gradient exists.”

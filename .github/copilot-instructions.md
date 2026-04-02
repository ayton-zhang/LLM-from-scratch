# LLM from Scratch — Workspace Guidelines

## Project Overview

A 9-part hands-on PyTorch curriculum building a complete LLM pipeline:
- **Parts 1–3**: Architecture (Transformer → GPT → Modern: RMSNorm, RoPE, SwiGLU, KV cache)
- **Parts 4–5**: Scaling (BPE, AMP, checkpointing, MoE)
- **Parts 6–9**: Alignment (SFT → Reward Model → PPO → GRPO)

## Build & Test

```bash
# Environment setup
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

**CRITICAL: All imports are local. Always run from inside the part directory.**

```bash
# Run all tests for a part
cd part_N
pytest -q

# Run a specific test
cd part_N
python -m pytest -q tests/test_foo.py

# Run the orchestrator (tests + demo)
cd part_N
python orchestrator.py          # tests only
python orchestrator.py --demo   # tests + generation demo
```

Docker alternative:
```bash
docker compose run llm-lab python part_1/demo_mha_shapes.py
```

## Architecture

### Key models
- **`part_2/model_gpt.py`** — Baseline GPT (byte tokenizer, learned positional embeddings, LayerNorm)
- **`part_3/model_modern.py`** — `GPTModern` (RMSNorm, RoPE, SwiGLU, KV cache). Used by parts 4–9.
- **`part_7/model_reward.py`** — Bidirectional Transformer encoder → scalar reward
- **`part_8/policy.py`, `part_9/policy.py`** — `PolicyWithValue` = GPTModern + value head

### Cross-part dependencies
- Parts 4, 6, 8, 9 import `GPTModern` from `part_3/` via `sys.path.append(..)`
- Parts 8–9 require checkpoints from parts 6 (SFT) and 7 (reward model)
- BPE tokenizer trained in part 4 is reused by parts 6–9 (`--bpe_dir ../part_4/runs/part4-demo/tokenizer`)

### Common model hyperparameter flags (all training scripts)
`--n_layer`, `--n_head`, `--n_embd`, `--block_size`, `--batch_size`, `--lr`

## Conventions

### Module structure per part
Each part contains:
- `orchestrator.py` — documents the layout (header comment) and runs tests + demo
- `tests/` — pytest unit tests; test only shapes, gradients, and mathematical invariants
- `runs/` — created at runtime; checkpoints saved here

### Testing pattern
Tests verify: output shapes, gradient flow (`p.grad is not None`), and mathematical properties (monotonicity, round-trips). They do **not** test convergence or generation quality.

### SFT prompt format (parts 6–9)
```
### Instruction:
{prompt}

### Response:
{response}
```
Labels for prompt tokens are masked to `-100`; loss is only on response tokens.

### Checkpoints
Saved as `{out_dir}/model_last.pt` with keys: `model` (state_dict), `config` (hyperparams dict). Load with:
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt['model'])
cfg = ckpt.get('config', {})
```

### Tokenizers
- Parts 1–3: byte-level (`vocab_size=256`)
- Parts 4–9: BPE via `tokenizer_bpe.BPETokenizer` (HuggingFace `tokenizers` under the hood)

## Pitfalls

- **Import errors**: Scripts must be run from inside their `part_N/` directory. Relative imports like `from model_modern import ...` will fail from the root.
- **Part 3 rope file**: The file is `rope_custom.py` (not `rope.py`) — orchestrator comment has a slight mismatch.
- **Part 9 orchestrator**: Header says "Run from inside `part_8/`" but it should be `part_9/`.
- **GPU access in Docker**: Requires NVIDIA runtime; `docker compose` config passes `count: all` GPUs.
- **`torch.compile`**: Prepends `_orig_mod.` to state dict keys; `part_2/train.py` strips this on save.

# How to Practice — CS336 Self-Study Guide

All `src/cs336/` modules are **skeleton-only**: docstrings + `raise NotImplementedError`.
You write every line of code yourself. The `lectures/` scripts are tutorial
material (educational, not assignment solutions).

## Where to start

### 1. Watch Lecture 1, then implement:
```
src/cs336/tokenizer/bpe.py     →  BPETokenizer.encode, .decode, run_train_bpe
```
Test: write a small script that trains BPE on a text file and encodes/decodes.

### 2. Watch Lectures 2-4, then implement (Assignment 1):
```
src/cs336/architecture/transformer.py
    ├── run_rmsnorm        (~5 lines)
    ├── run_silu           (~1 line)
    ├── run_linear         (~1 line)
    ├── run_embedding      (~1 line)
    ├── run_swiglu         (~6 lines)
    ├── run_scaled_dot_product_attention  (~10 lines)
    ├── run_rope           (~15 lines)   ← hardest individual function
    ├── run_multihead_self_attention       (~25 lines)
    ├── run_multihead_self_attention_with_rope  (~30 lines)
    ├── run_transformer_block              (~20 lines)
    └── run_transformer_lm                (~25 lines)

src/cs336/training/trainer.py
    ├── run_softmax                    (~5 lines)
    ├── run_cross_entropy              (~8 lines)
    ├── run_gradient_clipping          (~8 lines)
    ├── run_get_lr_cosine_schedule     (~12 lines)
    ├── AdamW class                    (~40 lines)
    ├── run_save_checkpoint            (~5 lines)
    ├── run_load_checkpoint            (~5 lines)
    └── run_get_batch                  (~12 lines)
```
Test: `pytest tests/` (write your own or use the official A1 tests)

### 3. Watch Lectures 5-8, then implement (Assignment 2):
```
src/cs336/systems/kernels.py
    ├── run_flash_attention2   (Triton kernel, ~80 lines, Linux only)
    ├── run_ddp                (~15 lines)
    └── run_fsdp               (~50 lines)
```

### 4. Watch Lectures 9-14 → `lectures/` scripts have concepts + exercises

### 5. Watch Lectures 15-17 → implement DPO, GRPO (Assignment 5)

## How the project helps

| What | cheat? | purpose |
|------|--------|---------|
| `src/cs336/*/` | **No** — all `raise NotImplementedError` | Your implementation canvas |
| `lectures/lecture_*.py` | No — tutorial demos, not solutions | Run to understand concepts |
| `ref/lectures/*.pdf` | No — official slides | Reference when stuck |
| `ref/assignments/*.pdf` | No — official handouts | Read for detailed specs |
| `.ref/` (gitignored) | No — full official repos | grep for hints if desperate |

## Running your code

```bash
cd ~/code/stanford_cs336
source .venv/bin/activate

# Quick test of one function
python -c "from cs336.architecture import run_silu; ..."

# Run the lecture tutorials (educational, not solutions)
python lectures/lecture_01_tokenization.py

# Write your own test scripts in tests/
python tests/test_my_tokenizer.py
```

## Advice from the course staff

- Debug on CPU first, use GPU only when needed
- Disable AI autocomplete (Copilot, Cursor Tab) — it hurts learning
- Study groups OK, but write your own code
- Don't look at existing implementations online unless the handout says to

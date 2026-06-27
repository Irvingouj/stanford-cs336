# How to Practice — CS336 Self-Study Guide

All `src/cs336/` modules are **skeleton-only**: docstrings + `raise NotImplementedError`.
You write every line yourself.

## What to read NOW vs LATER

| directory | safe now? | content |
|-----------|-----------|---------|
| `src/cs336/` | ✅ | Your canvas — all `raise NotImplementedError` |
| `lectures/` | ✅ | Conceptual demos, no assignment overlap |
| `references/` | ⚠️ **LATER** | Implementations that overlap with assignments — read after you finish |
| `ref/lectures/*.pdf` | ✅ | Official slide decks |
| `ref/assignments/*.pdf` | ✅ | Official assignment handouts |

## Recommended order

### Step 1: Watch Lecture 1 → implement tokenizer
```
src/cs336/tokenizer/bpe.py
    ├── BPETokenizer.encode()
    ├── BPETokenizer.decode()
    └── run_train_bpe()
```
When done, compare with: `references/lecture_01_tokenization.py`

### Step 2: Watch Lectures 2-4 → implement architecture (Assignment 1)
```
src/cs336/architecture/transformer.py
    ├── run_rmsnorm        (~5 lines)
    ├── run_silu           (~1 line)
    ├── run_linear         (~1 line)
    ├── run_embedding      (~1 line)
    ├── run_swiglu         (~6 lines)
    ├── run_scaled_dot_product_attention  (~10 lines)
    ├── run_rope           (~15 lines)   ← hardest
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
When done, compare with: `references/lecture_03_architectures.py`

### Step 3: Watch Lectures 5-8 → implement systems (Assignment 2)
```
src/cs336/systems/kernels.py
    ├── run_flash_attention2   (Triton kernel, ~80 lines, Linux only)
    ├── run_ddp                (~15 lines)
    └── run_fsdp               (~50 lines)
```
When done, compare with: `references/lecture_06_kernels_triton.py`

### Step 4: Lectures 9-14 → run the safe `lectures/` scripts for concepts
```
python lectures/lecture_09_11_scaling_laws.py
python lectures/lecture_10_inference.py
python lectures/lecture_12_evaluation.py
python lectures/lecture_13_14_data.py
```

### Step 5: Lectures 15-17 → implement alignment (Assignment 5)
When done, compare with: `references/lecture_15_16_training.py`

## Running your code

```bash
cd ~/code/stanford_cs336
source .venv/bin/activate

# Test one function
python -c "
import torch
from cs336.architecture import run_silu
# ... call your implementation
"

# Run safe conceptual demos
python lectures/lecture_02_pytorch_flops.py
python lectures/lecture_05_gpus_tpus.py

# Compare with reference AFTER you finish
python references/lecture_01_tokenization.py
```

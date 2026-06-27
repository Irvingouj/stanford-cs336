# Stanford CS336: Language Modeling from Scratch

**Spring 2026** — Percy Liang & Tatsunori Hashimoto

Build a complete language model from the ground up: data, tokenizer, transformer
architecture, GPU kernels, distributed training, scaling laws, evaluation, and
alignment (SFT/RLHF).

## Course Links

- [Course Website](https://stanford-cs336.github.io)
- [GitHub Organization](https://github.com/stanford-cs336)
- [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## Lecture Map

| # | Topic | Dir |
|---|-------|-----|
| 1 | Overview, Tokenization | `src/cs336/tokenizer/` |
| 2 | PyTorch (einops), FLOPs/memory | `src/cs336/architecture/` |
| 3 | Architectures, hyperparameters | `src/cs336/architecture/` |
| 4 | Attention alternatives, MoE | `src/cs336/architecture/` |
| 5 | GPUs, TPUs | `src/cs336/systems/` |
| 6 | Kernels, Triton | `src/cs336/systems/` |
| 7–8 | Parallelism | `src/cs336/systems/` |
| 9, 11 | Scaling laws | `src/cs336/scaling/` |
| 10 | Inference | `src/cs336/inference/` |
| 12 | Evaluation | `src/cs336/evaluation/` |
| 13–14 | Data — sources, filtering, dedup | `src/cs336/data/` |
| 15–16 | Mid/post-training (SFT/RLHF) | `src/cs336/training/` |
| 17 | Alignment, multimodality | `src/cs336/training/` |

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & sync
cd ~/code/stanford_cs336
uv sync

# With dev tools (jupyter, pytest, ruff)
uv sync --group dev

# Activate the venv
source .venv/bin/activate

# Launch Jupyter
jupyter lab
```

## GPU Notes

This course is compute-heavy. Debug on CPU first, then use a GPU:

- [Modal](https://modal.com) (course sponsor)
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)
- [Nebius](https://nebius.com)

Install Triton + FlashAttention (Linux only):

```bash
uv sync --group triton --group flash
```

## Assignments (from the official course)

1. **A1 Basics** — Tokenizer, transformer, optimizer, minimal LM training
2. **A2 Systems** — Profiling, Triton FlashAttention2, distributed training
3. **A3 Scaling** — Component analysis, scaling law fitting
4. **A4 Data** — Common Crawl → pretraining data pipeline
5. **A5 Alignment** — SFT + RL for math reasoning; safety with DPO

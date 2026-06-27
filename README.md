# Stanford CS336: Language Modeling from Scratch

**Spring 2026** — Percy Liang & Tatsunori Hashimoto

Build a complete language model from the ground up: data, tokenizer, transformer
architecture, GPU kernels, distributed training, scaling laws, evaluation, and
alignment (SFT/RLHF).

## Course Links

- [Course Website](https://stanford-cs336.github.io)
- [GitHub Organization](https://github.com/stanford-cs336)
- [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## Lectures (`.py` scripts)

Each lecture is a self-contained, runnable Python script:

| # | Script | Topic | Instructor |
|---|--------|-------|------------|
| 1 | `lecture_01_tokenization.py` | Overview, BPE tokenization | Percy |
| 2 | `lecture_02_pytorch_flops.py` | einops, FLOPs, memory, arithmetic intensity | Percy |
| 3 | `lecture_03_architectures.py` | GPT-2/LLaMA architecture, hyperparameters | Tatsu |
| 4 | `lecture_04_attention_moe.py` | Sliding window attention, Mixture of Experts | Tatsu |
| 5 | `lecture_05_gpus_tpus.py` | GPU/TPU architecture, memory hierarchy | Tatsu |
| 6 | `lecture_06_kernels_triton.py` | Triton kernels, FlashAttention | Percy |
| 7–8 | `lecture_07_08_parallelism.py` | Data/tensor/pipeline parallelism, 3D, ZeRO | Percy/Tatsu |
| 9,11 | `lecture_09_11_scaling_laws.py` | Kaplan, Chinchilla, compute-optimal scaling | Tatsu |
| 10 | `lecture_10_inference.py` | Decoding, KV-cache, quantization | Percy |
| 12 | `lecture_12_evaluation.py` | Perplexity, benchmarks, contamination | Percy |
| 13–14 | `lecture_13_14_data.py` | Data filtering, MinHash dedup, mixing | Percy |
| 15–16 | `lecture_15_16_training.py` | SFT, DPO, GRPO, reward models | Tatsu |
| 17 | `lecture_17_alignment_multimodal.py` | Constitutional AI, safety, multimodality | Percy |
| 18–19 | `lecture_18_19_guests.py` | Guest lectures (TBD) | Guests |

## Library Code (`src/cs336/`)

Reusable implementations organized by topic:

| Module | Lectures | Content |
|--------|----------|---------|
| `cs336.tokenizer` | 1 | BPE, SentencePiece, tiktoken wrappers |
| `cs336.architecture` | 2–4 | Transformer, attention, MoE, RoPE |
| `cs336.systems` | 5–8 | Triton kernels, parallelism utilities |
| `cs336.scaling` | 9,11 | Scaling law fitting, optimal allocation |
| `cs336.inference` | 10 | Decoding strategies, KV-cache |
| `cs336.evaluation` | 12 | Perplexity, benchmark harness |
| `cs336.data` | 13–14 | Data pipeline, dedup, filtering |
| `cs336.training` | 15–17 | SFT trainer, DPO, GRPO |

## Setup

```bash
# Clone
git clone https://github.com/Irvingouj/stanford-cs336.git
cd stanford-cs336

# Install dependencies
uv sync              # core (torch, einops, tiktoken, transformers, ...)
uv sync --group dev  # + jupyter, pytest, ruff, mypy

# Activate venv
source .venv/bin/activate

# Run a lecture
python lectures/lecture_01_tokenization.py
python lectures/lecture_03_architectures.py
```

## GPU Notes

This course is compute-heavy. Debug on CPU first, then use a GPU:

- [Modal](https://modal.com) (course sponsor)
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)
- [Nebius](https://nebius.com)

Triton & FlashAttention (Linux only):

```bash
uv sync --group triton --group flash
```

## Assignments (from the official course)

1. **A1 Basics** — Tokenizer, transformer, optimizer, minimal LM training
2. **A2 Systems** — Profiling, Triton FlashAttention2, distributed training
3. **A3 Scaling** — Component analysis, scaling law fitting
4. **A4 Data** — Common Crawl → pretraining data pipeline
5. **A5 Alignment** — SFT + RL for math reasoning; safety with DPO

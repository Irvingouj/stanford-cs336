# Stanford CS336: Language Modeling from Scratch

**Spring 2026** — Percy Liang & Tatsunori Hashimoto

Build a complete language model from the ground up.

## Links

- [Course Website](https://stanford-cs336.github.io)
- [GitHub Organization](https://github.com/stanford-cs336)
- [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## Structure

```
src/cs336/       ← implement these (all raise NotImplementedError)
reference/       ← complete reference (tutorial scripts + official PDFs)
```

## Setup

```bash
git clone https://github.com/Irvingouj/stanford-cs336.git
cd stanford-cs336
uv sync              # core deps (torch, einops, tiktoken, ...)
uv sync --group dev  # + jupyter, pytest, ruff, mypy
source .venv/bin/activate
```

## How to practice

Read **PRACTICE.md**. Two-step loop:

1. `python reference/lectures/lecture_XX_*.py` — understand the concept
2. Open `src/cs336/` — implement it yourself

## GPU compute (for self-study)

- [Modal](https://modal.com) — course sponsor, $30 free/month
- [Lambda Labs](https://lambdalabs.com) | [RunPod](https://runpod.io) | [Nebius](https://nebius.com)

Triton & FlashAttention (Linux only):
```bash
uv sync --group triton --group flash
```

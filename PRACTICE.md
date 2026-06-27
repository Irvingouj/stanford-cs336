# CS336 Practice Guide

Two folders. One goal.

## `src/cs336/` — Your canvas (implement these)

All `raise NotImplementedError`. You write every line.

```
src/cs336/
├── tokenizer/bpe.py          A1: BPE encode, decode, train
├── architecture/transformer.py  A1: RMSNorm → RoPE → SwiGLU → MHA → block → LM
├── systems/kernels.py        A2: FlashAttention2 (Triton), DDP, FSDP
└── training/trainer.py       A1: softmax, cross-entropy, AdamW, scheduling
```

## `reference/` — Complete reference (read anytime)

```
reference/
├── lectures/    14 tutorial .py scripts — working demos for every topic
└── official/    14 PDFs — 8 lecture slides + 6 assignment handouts
```

| Lecture | reference/lectures/ | reference/official/ |
|---------|---------------------|---------------------|
| 1 Tokenization | `lecture_01_tokenization.py` | assignment1 PDF |
| 2 PyTorch/FLOPs | `lecture_02_pytorch_flops.py` | — |
| 3 Architectures | `lecture_03_architectures.py` | `lecture_03.pdf` |
| 4 MoE | `lecture_04_attention_moe.py` | `lecture_04.pdf` |
| 5 GPUs/TPUs | `lecture_05_gpus_tpus.py` | `lecture_05.pdf` |
| 6 Triton | `lecture_06_kernels_triton.py` | assignment2 PDF |
| 7-8 Parallelism | `lecture_07_08_parallelism.py` | `lecture_08.pdf` |
| 9,11 Scaling | `lecture_09_11_scaling_laws.py` | `lecture_09.pdf`, `lecture_11.pdf` |
| 10 Inference | `lecture_10_inference.py` | assignment3 PDF |
| 12 Evaluation | `lecture_12_evaluation.py` | assignment4 PDF |
| 13-14 Data | `lecture_13_14_data.py` | — |
| 15-16 Training | `lecture_15_16_training.py` | `lecture_15.pdf`, `lecture_16.pdf`, assignment5 PDF |
| 17 Alignment | `lecture_17_alignment_multimodal.py` | — |
| 18-19 Guests | `lecture_18_19_guests.py` | — |

## Start here

```bash
source .venv/bin/activate

# 1. Read the concept
python reference/lectures/lecture_01_tokenization.py

# 2. Implement it
#    Open src/cs336/tokenizer/bpe.py and write your code

# 3. Check your work against the reference
python reference/lectures/lecture_01_tokenization.py
```

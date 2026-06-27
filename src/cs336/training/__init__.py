"""
Lectures 15-17: Mid/Post-Training & Alignment.

Topics:
- Lecture 15: Supervised fine-tuning (SFT), RLHF (PPO, DPO)
- Lecture 16: RLVR (RL with verifiable rewards) for math reasoning
- Lecture 17: Alignment tax, multimodality overview
"""

from cs336.training.trainer import (
    AdamW,
    get_adamw_cls,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
)

__all__ = [
    "AdamW",
    "get_adamw_cls",
    "run_get_batch",
    "run_get_lr_cosine_schedule",
    "run_gradient_clipping",
    "run_load_checkpoint",
    "run_save_checkpoint",
]

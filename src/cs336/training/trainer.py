"""
Training utilities — optimizer, learning rate schedule, gradient clipping,
checkpointing, data loading.

All functions raise NotImplementedError — implement them yourself.

Reference:
    https://github.com/stanford-cs336/assignment1-basics
    AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2017
"""

from __future__ import annotations

import io
import math
import os
import struct
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ── Gradient clipping ───────────────────────────────────────────────────


def run_gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients so their combined L2 norm is at most max_l2_norm.

    Modifies parameter.grad in-place. Skip parameters where .grad is None.

    total_norm = sqrt(sum(p.grad^2 for all p))
    if total_norm > max_l2_norm: scale all grads by max_l2_norm / total_norm

    Args:
        parameters: trainable parameters with .grad attributes
        max_l2_norm: maximum allowed L2 norm
    """
    raise NotImplementedError("TODO: implement gradient clipping")


# ── Cosine learning rate schedule ───────────────────────────────────────


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    - it < warmup_iters:               linear from 0 to max_learning_rate
    - warmup_iters <= it <= cosine_cycle_iters:  cosine decay to min_learning_rate
    - it > cosine_cycle_iters:          min_learning_rate

    Args:
        it: current iteration (0-indexed)
        max_learning_rate: peak LR (alpha_max)
        min_learning_rate: final LR (alpha_min)
        warmup_iters: warmup steps (T_w)
        cosine_cycle_iters: total cosine steps (T_c)

    Returns:
        learning rate at iteration `it`
    """
    raise NotImplementedError("TODO: implement cosine LR schedule")


# ── AdamW optimizer ─────────────────────────────────────────────────────


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implemented from scratch.

    AdamW decouples weight decay from gradient updates:

        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat   = m_t / (1 - beta1^t)
        v_hat   = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})

    Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2017.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        raise NotImplementedError("TODO: implement AdamW.__init__")

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("TODO: implement AdamW.step")


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return the AdamW optimizer class."""
    raise NotImplementedError("TODO: return AdamW class")


# ── Checkpointing ───────────────────────────────────────────────────────


def run_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model state, optimizer state, and iteration to disk.

    Hint: torch.save() a dict with keys 'model_state_dict',
    'optimizer_state_dict', 'iteration'.

    Args:
        model: the model
        optimizer: the optimizer
        iteration: current training step
        out: destination path or file-like object
    """
    raise NotImplementedError("TODO: implement save_checkpoint")


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model, optimizer, and iteration from a checkpoint.

    Args:
        src: checkpoint path or file-like object
        model: restore state into this model
        optimizer: restore state into this optimizer

    Returns:
        iteration number from the checkpoint
    """
    raise NotImplementedError("TODO: implement load_checkpoint")


# ── Data loading ────────────────────────────────────────────────────────


def run_get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of input sequences and labels from a 1D token array.

    For each sequence:
    - Pick a random start position
    - Input:  tokens[start : start + context_length]
    - Labels: tokens[start+1 : start + context_length + 1]
      (each label is the next token)

    Args:
        dataset: 1D numpy array of token IDs
        batch_size: number of sequences per batch
        context_length: sequence length
        device: torch device string

    Returns:
        (inputs, labels) — both (batch_size, context_length) LongTensors
    """
    raise NotImplementedError("TODO: implement get_batch")


# ── Cross-entropy loss ──────────────────────────────────────────────────


def run_cross_entropy(
    inputs: Tensor,  # (batch, vocab_size)
    targets: Tensor,  # (batch,)
) -> Tensor:
    """Compute average cross-entropy loss.

    cross_entropy = -mean(log(softmax(inputs)[target]))

    Args:
        inputs: unnormalized logits (batch, vocab_size)
        targets: correct class indices (batch,)

    Returns:
        scalar loss
    """
    raise NotImplementedError("TODO: implement cross-entropy loss")


# ── Softmax ─────────────────────────────────────────────────────────────


def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    """Compute softmax along the given dimension.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Be numerically stable: subtract max(x) before exp.
    """
    raise NotImplementedError("TODO: implement softmax")

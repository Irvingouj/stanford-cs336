"""
Training utilities — optimizer, learning rate schedule, gradient clipping.

Implements the interfaces expected by CS336 Assignment 1:
- get_adamw_cls(): AdamW optimizer from scratch
- run_get_lr_cosine_schedule(): cosine LR with linear warmup
- run_gradient_clipping(): gradient norm clipping
- run_save_checkpoint() / run_load_checkpoint(): serialization
"""

from __future__ import annotations

import io
import math
import os
import struct
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import torch
import torch.nn as nn
from torch import Tensor


# ── Gradient clipping ───────────────────────────────────────────────────


def run_gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients to have L2 norm at most max_l2_norm.

    Modifies parameter.grad in-place.
    """
    if max_l2_norm <= 0:
        return

    total_norm = torch.sqrt(
        sum(
            torch.sum(p.grad.detach() ** 2)
            for p in parameters
            if p.grad is not None
        )
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(scale)


# ── Cosine learning rate schedule ───────────────────────────────────────


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    - Linear warmup from 0 to max_lr over warmup_iters
    - Cosine decay from max_lr to min_lr over cosine_cycle_iters

    Args:
        it: current iteration
        max_learning_rate: peak LR (alpha_max)
        min_learning_rate: final LR (alpha_min)
        warmup_iters: warmup duration (T_w)
        cosine_cycle_iters: cosine annealing duration (T_c)

    Returns:
        learning rate at iteration it
    """
    if it < warmup_iters:
        # Linear warmup
        return max_learning_rate * (it + 1) / warmup_iters
    elif it <= cosine_cycle_iters:
        # Cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        return min_learning_rate


# ── AdamW optimizer ─────────────────────────────────────────────────────


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implemented from scratch.

    AdamW decouples weight decay from the gradient update:
        θ_{t+1} = θ_t - lr * (m_hat / (sqrt(v_hat) + eps) + wd * θ_t)

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
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # Update biased moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # AdamW update (decoupled weight decay)
                p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return the AdamW optimizer class."""
    return AdamW


# ── Checkpointing ───────────────────────────────────────────────────────


def run_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer state, and iteration to a checkpoint file.

    Args:
        model: the model to serialize
        optimizer: the optimizer to serialize
        iteration: current training iteration
        out: destination path or file-like object
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model, optimizer state, and iteration from a checkpoint.

    Args:
        src: checkpoint path or file-like object
        model: restore state into this model
        optimizer: restore state into this optimizer

    Returns:
        iteration number from the checkpoint
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


# ── Data loading ────────────────────────────────────────────────────────

import numpy as np


def run_get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of input sequences and labels from a 1D token dataset.

    Args:
        dataset: 1D numpy array of token IDs
        batch_size: number of sequences per batch
        context_length: sequence length
        device: torch device string

    Returns:
        (inputs, labels) — both (batch_size, context_length)
    """
    # Random starting positions
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = np.stack([
        dataset[start:start + context_length] for start in starts
    ])
    labels = np.stack([
        dataset[start + 1:start + context_length + 1] for start in starts
    ])

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )

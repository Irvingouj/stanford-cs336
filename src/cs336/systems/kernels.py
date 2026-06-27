"""
Systems — FlashAttention, DDP, FSDP.

Implements interfaces from CS336 Assignment 2.

All functions raise NotImplementedError — implement them yourself.

Reference:
    https://github.com/stanford-cs336/assignment2-systems
    FlashAttention: Dao et al., https://arxiv.org/abs/2205.14135
    FlashAttention-2: Dao, https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor


# ── FlashAttention 2 ────────────────────────────────────────────────────


def run_flash_attention2(
    Q: Tensor,  # (batch, n_heads, seq_len, head_dim)
    K: Tensor,
    V: Tensor,
    causal: bool = True,
) -> Tensor:
    """
    FlashAttention 2: IO-aware exact attention with O(N) memory.

    Key ideas:
    - Tile Q into blocks, iterate over K/V blocks
    - Maintain online softmax statistics (m, l) per Q block
    - Fuse operations in a single Triton/CUDA kernel
    - Avoid materializing the full N×N attention matrix

    For the real Triton implementation:
    1. Compute Q @ K^T in tiles
    2. Online softmax: update running max and sum per Q-block
    3. Accumulate weighted V in tiles
    4. Normalize at the end

    Triton is Linux-only. Install: uv sync --group triton

    Args:
        Q: queries (batch, n_heads, seq_len, head_dim)
        K: keys
        V: values
        causal: whether to apply causal masking

    Returns:
        attention output (batch, n_heads, seq_len, head_dim)
    """
    raise NotImplementedError("TODO: implement FlashAttention2 in Triton")


# ── Distributed Data Parallel ───────────────────────────────────────────


def run_ddp(
    model: nn.Module,
    local_batch: tuple[Tensor, Tensor],
    world_size: int,
    rank: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    One DDP training step.

    1. Forward pass on local micro-batch
    2. Compute loss
    3. Backward pass
    4. All-reduce gradients across ranks (use dist.all_reduce with AVG)
    5. Return loss and gradients

    Args:
        model: the model (each rank has a full copy)
        local_batch: (inputs, labels) for this rank
        world_size: total number of GPUs
        rank: this GPU's rank

    Returns:
        (loss, grad_dict) — averaged loss and gradients
    """
    raise NotImplementedError("TODO: implement DDP training step")


# ── FSDP (Fully Sharded Data Parallel) ──────────────────────────────────


def run_fsdp(
    model: nn.Module,
    local_batch: tuple[Tensor, Tensor],
    world_size: int,
    rank: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    One FSDP training step (ZeRO-3 style).

    Key differences from DDP:
    - Parameters, gradients, and optimizer states are SHARDED across GPUs
    - All-gather parameters before forward (dist.all_gather)
    - Forward with full parameters
    - Discard full parameters to free memory
    - Backward: compute grad w.r.t. full parameters
    - Reduce-scatter gradients (dist.reduce_scatter) — only keep your shard

    Pseudocode for one layer:
        1. all_gather(sharded_W) → full_W
        2. forward with full_W
        3. discard full_W
        4. backward: grad w.r.t. full_W
        5. reduce_scatter(full_grad) → sharded_grad (keep only your piece)

    Args:
        model: the model (parameters sharded across ranks)
        local_batch: (inputs, labels) for this rank
        world_size: total GPUs
        rank: local rank

    Returns:
        (loss, grad_dict)
    """
    raise NotImplementedError("TODO: implement FSDP all-gather/reduce-scatter")

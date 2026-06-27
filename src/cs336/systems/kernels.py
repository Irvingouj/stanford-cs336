"""
Systems — FlashAttention, DDP, FSDP.

Implements interfaces from CS336 Assignment 2:
- run_flash_attention2(): Triton FlashAttention2 kernel
- run_ddp(): DistributedDataParallel training step
- run_fsdp(): FullyShardedDataParallel wrapping

Reference:
    https://github.com/stanford-cs336/assignment2-systems
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
    - Fuse all operations in a single Triton/CUDA kernel
    - Avoid materializing the full NxN attention matrix

    For Triton implementation, see the lecture_06 script and
    the official assignment handout.

    Args:
        Q: queries (batch, n_heads, seq_len, head_dim)
        K: keys
        V: values
        causal: whether to apply causal masking

    Returns:
        attention output (batch, n_heads, seq_len, head_dim)
    """
    # TODO: implement with Triton kernel
    # Fallback: use PyTorch's scaled_dot_product_attention (cuDNN FA backend)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=causal,
    )


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
    4. All-reduce gradients across ranks
    5. Return loss and gradients

    Args:
        model: the model (each rank has a full copy)
        local_batch: (inputs, labels) for this rank
        world_size: total number of GPUs
        rank: this GPU's rank

    Returns:
        (loss, grad_dict) — averaged loss and gradients
    """
    inputs, labels = local_batch

    # Forward
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )

    # Backward
    loss.backward()

    # All-reduce gradients
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    return loss, grads


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
    - Parameters, gradients, and optimizer states are sharded across GPUs
    - All-gather parameters before forward pass
    - Reduce-scatter gradients after backward pass
    - Discard full parameters to free memory

    Args:
        model: the model (wrapped with FSDP)
        local_batch: (inputs, labels) for this rank
        world_size: total GPUs
        rank: local rank

    Returns:
        (loss, grad_dict)
    """
    # FSDP is typically handled by PyTorch's FSDP wrapper or DeepSpeed ZeRO-3.
    # Students implement the all-gather / reduce-scatter logic manually.
    #
    # Pseudocode for one layer:
    # 1. all_gather(sharded_W) → full_W
    # 2. forward with full_W
    # 3. discard full_W
    # 4. backward: compute grad w.r.t. full_W
    # 5. reduce_scatter(full_grad) → sharded_grad  # only keep your shard
    #
    raise NotImplementedError("TODO: implement FSDP all-gather/reduce-scatter")

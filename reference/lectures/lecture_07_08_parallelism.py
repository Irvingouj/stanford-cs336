"""
Stanford CS336 — Lectures 7 & 8: Parallelism
==============================================
Spring 2026. Instructors: Percy Liang, Tatsu Hashimoto.

Topics:
- Data parallelism (DP, DDP, FSDP)
- Tensor parallelism (Megatron-style)
- Pipeline parallelism (GPipe)
- ZeRO (DeepSpeed stages 1/2/3)
- 3D parallelism: combining all three
- Communication primitives: all-reduce, all-gather, reduce-scatter

These are PDF lectures in the official repo. This script provides
code sketches of the core ideas.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


# ── Data Parallelism ────────────────────────────────────────────────────

def data_parallel_sketch():
    """
    Data Parallelism (DDP):
    - Each GPU has a full copy of the model
    - Each GPU processes a different micro-batch
    - Gradients are all-reduced across GPUs after backward()

    Pros: simple, no model changes
    Cons: each GPU must fit full model + optimizer states
    """
    print("""
    ═══════════ Data Parallelism (DDP) ═══════════
    GPU 0: [model full copy]  batch[0:8]   → grad_0
    GPU 1: [model full copy]  batch[8:16]  → grad_1
    GPU 2: [model full copy]  batch[16:24] → grad_2
    GPU 3: [model full copy]  batch[24:32] → grad_3

    After backward: all-reduce(grad_0, grad_1, grad_2, grad_3)
    Each GPU updates its own copy: θ -= lr * avg_grad

    ZeRO stages reduce memory:
    - ZeRO-1: shard optimizer states
    - ZeRO-2: shard gradients + optimizer states
    - ZeRO-3: shard parameters + gradients + optimizer states
    (FSDP = PyTorch's ZeRO-3)
    """)


# ── Tensor Parallelism ──────────────────────────────────────────────────

class ColumnParallelLinear(nn.Module):
    """
    Tensor parallelism: split a weight matrix across GPUs by column.

    Y = X @ W   →   Y = [X @ W_0 | X @ W_1]

    Each GPU holds a slice of W and computes part of the output.
    """
    def __init__(self, in_features: int, out_features: int, world_size: int = 4):
        super().__init__()
        self.world_size = world_size
        self.in_features = in_features
        self.out_per_gpu = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_per_gpu) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_features)
        # weight: (in_features, out_per_gpu)
        return x @ self.weight  # (batch, seq, out_per_gpu)


class RowParallelLinear(nn.Module):
    """
    Row parallelism: split by row, all-reduce the result.

    Y = X @ W   →   Y = X_0 @ W_0 + X_1 @ W_1 (all-reduce after)
    """
    def __init__(self, in_features: int, out_features: int, world_size: int = 4):
        super().__init__()
        self.world_size = world_size
        self.in_per_gpu = in_features // world_size
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(self.in_per_gpu, out_features) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already split: (batch, seq, in_per_gpu)
        return x @ self.weight  # (batch, seq, out_features)
        # In practice: all-reduce the outputs across GPUs


# ── Pipeline Parallelism ────────────────────────────────────────────────

def pipeline_parallel_sketch():
    """Print the pipeline parallelism explanation."""
    explanation = (
        "\n    Pipeline Parallelism (GPipe):\n"
        "    - Split model layers across GPUs\n"
        "    - Each GPU holds a contiguous chunk of layers\n"
        "    - Micro-batches pipeline through GPUs\n"
        "\n"
        '    Key challenge: "bubble" — GPUs idle waiting for dependencies.\n'
        "    Solution: more micro-batches to fill the pipeline.\n"
        "\n"
        "    Pipeline (4 GPUs, 8 micro-batches):\n"
        "    Time ->\n"
        "    GPU 0: [M0][M1][M2][M3][M4][M5][M6][M7]............\n"
        "    GPU 1: ....[M0][M1][M2][M3][M4][M5][M6][M7]........\n"
        "    GPU 2: ........[M0][M1][M2][M3][M4][M5][M6][M7]....\n"
        "    GPU 3: ............[M0][M1][M2][M3][M4][M5][M6][M7]\n"
        "\n"
        "    Bubble fraction ~ (p-1) / (p-1 + m)  where p=GPUs, m=micro-batches\n"
        "    More micro-batches -> less bubble -> more memory for activations\n"
    )
    print(explanation)


# ── 3D Parallelism ──────────────────────────────────────────────────────

def three_d_parallelism():
    """
    3D Parallelism = Data + Tensor + Pipeline

    Example (training a 175B model on 1024 GPUs):
    - DP=64   (data parallel replicas)
    - TP=8    (tensor parallel within a node, NVLink)
    - PP=4    (pipeline stages)

    64 × 8 × 4 = 2048 ... wait, that's 2048. For 1024 GPUs:
    - DP=32, TP=8, PP=4  →  32 × 8 × 4 = 1024

    Communication costs:
    - DP: all-reduce gradients (inter-node, network-bound)
    - TP: all-reduce activations (intra-node, NVLink/NVSwitch)
    - PP: send/recv activations (point-to-point, inter-node)
    """
    print(three_d_parallelism.__doc__)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_parallel_sketch()
    pipeline_parallel_sketch()
    three_d_parallelism()

    print("\n── ColumnParallelLinear test ──\n")
    col_lin = ColumnParallelLinear(in_features=512, out_features=512, world_size=4)
    x = torch.randn(2, 64, 512)
    y = col_lin(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}  (each GPU holds 128 of 512 cols)")

    print("\n── RowParallelLinear test ──\n")
    row_lin = RowParallelLinear(in_features=512, out_features=512, world_size=4)
    x_split = torch.randn(2, 64, 128)  # already sharded
    y = row_lin(x_split)
    print(f"Input:  {x_split.shape}  (each GPU gets 128 of 512 input dims)")
    print(f"Output: {y.shape}  (full output, needs all-reduce)")

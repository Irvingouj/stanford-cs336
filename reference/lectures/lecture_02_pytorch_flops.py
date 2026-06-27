"""
Stanford CS336 — Lecture 2: PyTorch (einops), FLOPs, Memory, Arithmetic Intensity
===================================================================================
Spring 2026. Instructor: Percy Liang.

This lecture covers the PyTorch fundamentals you need for building
language models efficiently:
- einops: tensor operations made readable
- FLOPs accounting: how many operations does your model do?
- Memory footprint: how much GPU memory does training need?
- Arithmetic intensity: FLOPs / bytes — are you compute-bound or memory-bound?

References:
- https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py
- einops docs: https://einops.rocks
"""

import torch
import einops


# ── einops basics ───────────────────────────────────────────────────────

def einops_demo():
    """Core einops operations used throughout the course."""
    print("── einops: readable tensor operations ──\n")

    # --- rearrange: reshape + transpose in one call ---
    x = torch.randn(2, 3, 4)  # (batch, seq, hidden)
    print(f"x.shape: {x.shape}")

    # Transpose last two dims
    y = einops.rearrange(x, "b s h -> b h s")
    print(f"rearrange(x, 'b s h -> b h s').shape: {y.shape}")  # (2, 4, 3)

    # Merge dimensions
    y = einops.rearrange(x, "b s h -> (b s) h")
    print(f"rearrange(x, 'b s h -> (b s) h').shape: {y.shape}")  # (6, 4)

    # Split dimensions (on a tensor with matching shape)
    x2 = torch.randn(2, 6, 4)  # seq=6 = 2*3
    y = einops.rearrange(x2, "b (s1 s2) h -> b s1 s2 h", s1=2)
    print(f"split seq dim: {y.shape}")  # (2, 2, 3, 4)

    print()

    # --- repeat: broadcast along new axes ---
    x = torch.randn(3, 4)
    y = einops.repeat(x, "h w -> h w c", c=5)
    print(f"repeat(x, 'h w -> h w c', c=5).shape: {y.shape}")  # (3, 4, 5)

    # --- reduce: named reductions ---
    x = torch.randn(2, 3, 4)
    y = einops.reduce(x, "b s h -> b h", "mean")
    print(f"reduce(x, 'b s h -> b h', 'mean').shape: {y.shape}")  # (2, 4)

    print()


# ── FLOPs accounting ────────────────────────────────────────────────────

def flops_accounting():
    """
    Estimate FLOPs for transformer components.

    Key formulas (for a single token):
    - Attention:   4 * d_model^2 + 2 * n_ctx * d_model  (per layer, per token)
    - MLP:         8 * d_model^2  (per layer, per token, 4x expansion)

    For a full forward pass with N tokens, L layers:
        FLOPs ≈ 2 * N * L * d_model^2 * (4 + 8)   [rough]
               = 24 * N * L * d_model^2

    Backward pass is ~2× forward pass, so total:
        FLOPs_total ≈ 72 * N * L * d_model^2
    """
    print("── FLOPs accounting ──\n")

    d_model = 4096
    n_ctx = 2048
    n_layers = 32

    # Per-token, per-layer
    attn_flops = 4 * d_model**2 + 2 * n_ctx * d_model
    mlp_flops = 8 * d_model**2
    per_token_per_layer = attn_flops + mlp_flops

    print(f"d_model={d_model}, n_ctx={n_ctx}, n_layers={n_layers}")
    print(f"  Attention FLOPs (per token, per layer): {attn_flops:.2e}")
    print(f"  MLP FLOPs (per token, per layer):       {mlp_flops:.2e}")
    print(f"  Total per token per layer:              {per_token_per_layer:.2e}")

    # Full forward pass
    total_fwd = n_ctx * n_layers * per_token_per_layer
    total_bwd = 2 * total_fwd
    total = total_fwd + total_bwd

    print(f"  Forward pass (single seq):  {total_fwd:.2e} FLOPs = {total_fwd / 1e12:.2f} TFLOPs")
    print(f"  Forward + backward:         {total:.2e} FLOPs = {total / 1e12:.2f} TFLOPs")

    # For comparison: A100 FP16 = 312 TFLOPs/s
    a100_tflops = 312
    print(f"  A100 time (ideal, 312 TFLOPs): {total / (a100_tflops * 1e12):.4f} s")


# ── Memory footprint ────────────────────────────────────────────────────

def memory_footprint():
    """Estimate GPU memory needed for training."""
    print("\n── Memory footprint ──\n")

    d_model = 4096
    n_layers = 32
    n_ctx = 2048
    batch_size = 8
    bytes_per_param = 2  # FP16

    # Parameters (rough: 12 * d_model^2 per layer for attention + MLP)
    params_per_layer = 12 * d_model**2
    total_params = n_layers * params_per_layer
    param_memory = total_params * bytes_per_param

    print(f"Parameters: {total_params:.2e} ({total_params / 1e9:.2f}B)")
    print(f"  Parameter memory (FP16): {param_memory / 1e9:.2f} GB")

    # Activations (rough)
    # Attention: batch * n_heads * n_ctx * n_ctx  (FP16)
    # MLP:       batch * n_ctx * 4*d_model  (FP16)
    n_heads = 32
    head_dim = d_model // n_heads
    attn_act = batch_size * n_layers * n_heads * n_ctx * n_ctx * bytes_per_param
    mlp_act = batch_size * n_layers * n_ctx * 4 * d_model * bytes_per_param
    total_act = attn_act + mlp_act

    print(f"  Activation memory: ~{total_act / 1e9:.2f} GB")

    # Optimizer states (Adam: param + 2 moments)
    opt_memory = param_memory * 2  # fp32 moments ~= 2x fp16 params
    print(f"  Optimizer states (Adam): ~{opt_memory / 1e9:.2f} GB")

    total_memory = param_memory + total_act + opt_memory
    print(f"  Total estimated: ~{total_memory / 1e9:.2f} GB")

    # A100 has 40 GB or 80 GB
    print(f"  Fits A100-80GB: {'YES' if total_memory < 80e9 else 'NO'}")
    print(f"  Fits A100-40GB: {'YES' if total_memory < 40e9 else 'NO'}")


# ── Arithmetic intensity ────────────────────────────────────────────────

def arithmetic_intensity():
    """
    Arithmetic intensity = FLOPs / bytes_moved.

    - High intensity (> ~100): compute-bound → want more FLOPS
    - Low intensity (< ~100):  memory-bound → want more bandwidth

    For transformers:
    - Attention with small d_model: memory-bound
    - Large matrix multiplies (MLP): compute-bound
    - This is why FlashAttention matters — it reduces memory movement
    """
    print("\n── Arithmetic intensity ──\n")

    # A100 specs
    peak_flops = 312e12  # 312 TFLOPs (FP16)
    peak_bw = 2.0e12     # 2 TB/s (HBM2e)

    # Matmul: (M, K) x (K, N)
    # FLOPs ≈ 2 * M * N * K
    # Bytes  ≈ 2 * (M*K + K*N + M*N)
    for M, N, K in [(1024, 4096, 4096), (64, 64, 64), (2048, 2048, 8192)]:
        flops = 2 * M * N * K
        bytes_moved = 2 * (M * K + K * N + M * N)  # FP16
        ai = flops / bytes_moved
        bound = "compute-bound" if ai > peak_flops / peak_bw else "memory-bound"
        print(f"  matmul({M}×{K}) × ({K}×{N}): "
              f"FLOPs={flops:.2e}, bytes={bytes_moved:.2e}, "
              f"AI={ai:.1f} → {bound}")

    print(f"\n  A100 ridge point: {peak_flops / peak_bw:.1f} FLOPs/byte")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    einops_demo()
    flops_accounting()
    memory_footprint()
    arithmetic_intensity()

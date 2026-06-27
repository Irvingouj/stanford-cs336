"""
Stanford CS336 — Lecture 6: Kernels & Triton
==============================================
Spring 2026. Instructor: Percy Liang.

Topics:
- Writing custom GPU kernels in Triton
- Fused kernels: why and when
- FlashAttention: the canonical example
- Matrix multiplication in Triton

Triton is a Python DSL that compiles to CUDA. It lets you write
GPU kernels without writing CUDA C++.

Prerequisites:
    pip install triton  (Linux only)

Reference:
- https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py
- https://triton-lang.org
"""

import torch
import torch.nn.functional as F


# ── Pure PyTorch attention (for comparison) ─────────────────────────────

def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Standard scaled dot-product attention in PyTorch."""
    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return attn @ v


# ── FlashAttention-inspired: block-wise attention ────────────────────────

def block_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_size: int = 128,
):
    """
    Simplified block-wise attention to illustrate the FlashAttention idea.

    Instead of materializing the full (N, N) attention matrix:
    1. Split Q into blocks
    2. For each Q-block, iterate over K,V blocks
    3. Maintain running softmax statistics
    4. Only O(N*d) memory instead of O(N^2)

    This is NOT a real FlashAttention kernel — it's a PyTorch-level
    illustration. Real FlashAttention uses Triton/CUDA for fusion.
    """
    B, H, N, D = q.shape
    scale = D ** -0.5
    output = torch.zeros_like(q)

    for q_start in range(0, N, block_size):
        q_end = min(q_start + block_size, N)
        q_block = q[:, :, q_start:q_end]  # (B, H, Bs, D)

        # Running softmax stats (online softmax)
        m = torch.full((B, H, q_end - q_start, 1), float("-inf"),
                       device=q.device, dtype=q.dtype)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(q_block)

        for kv_start in range(0, N, block_size):
            kv_end = min(kv_start + block_size, N)
            k_block = k[:, :, kv_start:kv_end]  # (B, H, Bs, D)
            v_block = v[:, :, kv_start:kv_end]

            # Compute attention scores for this block
            scores = (q_block @ k_block.transpose(-2, -1)) * scale  # (B, H, q_bs, kv_bs)

            # Online softmax update
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            l_new = torch.exp(m - m_new) * l + torch.exp(scores - m_new).sum(dim=-1, keepdim=True)
            acc = torch.exp(m - m_new) * acc + torch.exp(scores - m_new) @ v_block

            m = m_new
            l = l_new

        # Normalize
        output[:, :, q_start:q_end] = acc / l

    return output


# ── Triton matmul stub ──────────────────────────────────────────────────

def triton_matmul_example():
    """
    If Triton were available, here's what a tiled matmul kernel looks like.
    This is non-functional — it shows the structure.

    Real Triton kernels are defined as:

        @triton.jit
        def matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(0)
            # ... tiled matmul logic ...
    """
    print("Triton is Linux-only. Install with:  uv sync --group triton")
    print("Then see official lecture_06.py for working Triton kernels.")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test block-wise attention correctness
    print("── Block-wise attention vs PyTorch ──\n")

    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    ref = pytorch_attention(q, k, v)
    out = block_attention(q, k, v, block_size=64)

    error = (ref - out).abs().max().item()
    print(f"  Shapes: ref={ref.shape}, out={out.shape}")
    print(f"  Max absolute error: {error:.6f}")
    print(f"  Match: {'PASS' if error < 1e-3 else 'FAIL — increase block_size or check numeric stability'}")

    # Memory comparison
    attn_mem = B * H * N * N * 2  # FP16 attention matrix in bytes
    block_mem = B * H * 64 * D * 2  # block-sized accumulator
    print(f"\n  Full attention memory: {attn_mem / 1024:.0f} KB")
    print(f"  Block-wise memory:     {block_mem / 1024:.0f} KB")
    print(f"  Reduction:             {attn_mem / block_mem:.0f}×")

    print("\n── Triton kernel note ──")
    triton_matmul_example()

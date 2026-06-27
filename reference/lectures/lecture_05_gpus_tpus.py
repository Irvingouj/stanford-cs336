"""
Stanford CS336 — Lecture 5: GPUs & TPUs
=========================================
Spring 2026. Instructor: Tatsu Hashimoto.

Topics:
- GPU architecture: SMs, warps, memory hierarchy
- CUDA programming model: threads, blocks, grids
- GPU memory: global, shared, registers, L1/L2 cache
- TPU architecture and systolic arrays
- How this matters for language model training

Reference:
- CUDA C++ Programming Guide
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
"""

# L5 is a PDF lecture. This script provides conceptual notes and a
# PyTorch-level demonstration of GPU memory hierarchy effects.


import torch


def memory_hierarchy_demo():
    """Demonstrate how GPU memory hierarchy affects performance."""
    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU. Install on a GPU machine to see effects.")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # Memory hierarchy (A100):
    # - Registers:     ~256 KB per SM,  ~64K × 32-bit registers
    # - Shared mem:    ~164 KB per SM (configurable with L1)
    # - L1 cache:      ~128 KB per SM (shared with shared mem)
    # - L2 cache:      40 MB (unified across all SMs)
    # - Global (HBM):  40 GB or 80 GB

    print("\n── Matrix multiply at different sizes ──")
    print("Small matmuls use registers/shared memory; large ones stress HBM.\n")

    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
             (4096, 4096, 4096), (8192, 8192, 8192)]

    for M, N, K in sizes:
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)

        # Warmup
        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()

        # Time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = a @ b
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 100  # ms per matmul

        flops = 2 * M * N * K
        tflops_per_s = flops / (elapsed / 1000) / 1e12
        print(f"  {M}×{K} × {K}×{N}:  {elapsed:.3f} ms  ({tflops_per_s:.2f} TFLOPs/s)")


if __name__ == "__main__":
    memory_hierarchy_demo()

    print("""
    GPU Memory Hierarchy (A100):
    ═══════════════════════════════════════════
    Level          Size         Latency
    ───────────────────────────────────────────
    Registers      ~256 KB/SM   ~0 cycles
    Shared/L1      ~192 KB/SM   ~30 cycles
    L2 Cache       40 MB        ~200 cycles
    HBM (Global)   40/80 GB     ~400 cycles
    ═══════════════════════════════════════════

    Key takeaway for LM training:
    - Attention is memory-bound (lots of data movement relative to compute)
    - Large matmuls (MLP layers) are compute-bound
    - FlashAttention fuses attention ops to avoid writing to HBM
    """)

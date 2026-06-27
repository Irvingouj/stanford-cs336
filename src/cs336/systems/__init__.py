"""
Lectures 5-8: Systems.

Topics:
- Lecture 5: GPU/TPU architecture, memory hierarchy, CUDA basics
- Lecture 6: Writing custom kernels in Triton
- Lectures 7-8: Data parallelism, tensor parallelism, pipeline
  parallelism, FSDP, ZeRO
"""

from cs336.systems.kernels import run_ddp, run_flash_attention2, run_fsdp

__all__ = ["run_ddp", "run_flash_attention2", "run_fsdp"]

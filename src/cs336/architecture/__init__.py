"""
Lectures 2-4: Architectures.

Topics:
- Lecture 2: PyTorch fundamentals (einops), FLOPs accounting,
  memory footprint, arithmetic intensity
- Lecture 3: Transformer architectures, hyperparameters,
  pre-norm vs post-norm, RoPE
- Lecture 4: Attention alternatives (linear, sliding window),
  mixture of experts (MoE)
"""

from cs336.architecture.transformer import (
    RMSNorm,
    SwiGLU,
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
)

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "run_embedding",
    "run_linear",
    "run_multihead_self_attention",
    "run_multihead_self_attention_with_rope",
    "run_rmsnorm",
    "run_rope",
    "run_scaled_dot_product_attention",
    "run_silu",
    "run_swiglu",
    "run_transformer_block",
    "run_transformer_lm",
]

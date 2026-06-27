"""
Transformer architecture components — from scratch.

Implements the interfaces expected by CS336 Assignment 1:
- RMSNorm, SiLU, SwiGLU
- Scaled dot-product attention
- Multi-head self-attention (with and without RoPE)
- Transformer block
- Transformer LM (full model)

All functions raise NotImplementedError — implement them yourself.

Reference:
    https://github.com/stanford-cs336/assignment1-basics
    https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Normalization ──────────────────────────────────────────────────────


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Tensor,  # (d_model,)
    in_features: Tensor,  # (..., d_model)
) -> Tensor:
    """RMSNorm: normalize by root-mean-square, then apply affine transform.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Args:
        d_model: feature dimension
        eps: numerical stability
        weights: learnable scale parameter (d_model,)
        in_features: input tensor (..., d_model)

    Returns:
        normalized tensor (..., d_model)
    """
    raise NotImplementedError("TODO: implement RMSNorm")


class RMSNorm(nn.Module):
    """RMSNorm as a nn.Module (used in LLaMA)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: implement RMSNorm.forward")


# ── Activation ─────────────────────────────────────────────────────────


def run_silu(in_features: Tensor) -> Tensor:
    """SiLU (Sigmoid Linear Unit) = x * sigmoid(x)."""
    raise NotImplementedError("TODO: implement SiLU")


# ── Linear ─────────────────────────────────────────────────────────────


def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,  # (d_out, d_in)
    in_features: Tensor,  # (..., d_in)
) -> Tensor:
    """Apply a linear transformation: out = in_features @ weights^T.

    Args:
        d_in: input feature dimension
        d_out: output feature dimension
        weights: weight matrix (d_out, d_in)
        in_features: input (..., d_in)

    Returns:
        output (..., d_out)
    """
    raise NotImplementedError("TODO: implement linear transformation")


# ── Embedding ──────────────────────────────────────────────────────────


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,  # (vocab_size, d_model)
    token_ids: Tensor,  # (...)
) -> Tensor:
    """Look up token embeddings from a weight matrix.

    Args:
        vocab_size: number of embeddings
        d_model: embedding dimension
        weights: embedding matrix (vocab_size, d_model)
        token_ids: token indices (...)

    Returns:
        embeddings (..., d_model)
    """
    raise NotImplementedError("TODO: implement embedding lookup")


# ── SwiGLU ─────────────────────────────────────────────────────────────


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,  # (d_ff, d_model)
    w2_weight: Tensor,  # (d_model, d_ff)
    w3_weight: Tensor,  # (d_ff, d_model)
    in_features: Tensor,  # (..., d_model)
) -> Tensor:
    """SwiGLU: Swish-Gated Linear Unit.

    SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T

    Args:
        d_model: input/output dimension
        d_ff: hidden dimension
        w1_weight: gate projection (d_ff, d_model)
        w2_weight: output projection (d_model, d_ff)
        w3_weight: up projection (d_ff, d_model)
        in_features: input (..., d_model)

    Returns:
        output (..., d_model)
    """
    raise NotImplementedError("TODO: implement SwiGLU")


class SwiGLU(nn.Module):
    """SwiGLU as a nn.Module."""
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: implement SwiGLU.forward")


# ── Scaled Dot-Product Attention ───────────────────────────────────────


def run_scaled_dot_product_attention(
    Q: Tensor,  # (..., queries, d_k)
    K: Tensor,  # (..., keys, d_k)
    V: Tensor,  # (..., keys, d_v)
    mask: Tensor | None = None,  # (..., queries, keys) or None
) -> Tensor:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

    The mask is bool: True = masked position (set to -inf before softmax).

    Args:
        Q: queries
        K: keys
        V: values
        mask: optional attention mask

    Returns:
        attention output (..., queries, d_v)
    """
    raise NotImplementedError("TODO: implement scaled dot-product attention")


# ── RoPE (Rotary Position Embedding) ───────────────────────────────────


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,  # (..., seq_len, d_k)
    token_positions: Tensor,  # (..., seq_len)
) -> Tensor:
    """Apply Rotary Position Embedding (RoPE).

    RoPE encodes position by rotating pairs of feature dimensions.
    angle_i = position / theta^(2i/d_k)

    Hint: treat adjacent dimension pairs as (real, imag) and rotate them.

    Args:
        d_k: head dimension
        theta: base frequency (e.g., 10000.0)
        max_seq_len: max sequence length (for pre-computation)
        in_query_or_key: Q or K tensor (..., seq_len, d_k)
        token_positions: position indices (..., seq_len)

    Returns:
        RoPE-transformed tensor (..., seq_len, d_k)
    """
    raise NotImplementedError("TODO: implement RoPE")


# ── Multi-Head Self-Attention ──────────────────────────────────────────


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,  # (d_model, d_model)
    k_proj_weight: Tensor,  # (d_model, d_model)
    v_proj_weight: Tensor,  # (d_model, d_model)
    o_proj_weight: Tensor,  # (d_model, d_model)
    in_features: Tensor,  # (..., seq_len, d_model)
) -> Tensor:
    """Multi-head self-attention (WITHOUT RoPE).

    Steps:
    1. Project Q, K, V from in_features (one matmul each, for all heads)
    2. Reshape to separate heads: (..., num_heads, seq_len, head_dim)
    3. Apply causal mask
    4. Call run_scaled_dot_product_attention
    5. Reassemble heads and apply output projection

    Args:
        d_model: model dimension
        num_heads: number of attention heads (d_model % num_heads == 0)
        q/k/v/o_proj_weight: projection weights
        in_features: input (..., seq_len, d_model)

    Returns:
        attention output (..., seq_len, d_model)
    """
    raise NotImplementedError("TODO: implement multi-head self-attention")


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,  # (..., seq_len, d_model)
    token_positions: Tensor | None = None,  # (..., seq_len)
) -> Tensor:
    """Multi-head self-attention WITH RoPE.

    Same as run_multihead_self_attention, but apply run_rope to Q and K
    after projection and before attention.

    If token_positions is None, create positions 0..seq_len-1.
    """
    raise NotImplementedError("TODO: implement multi-head self-attention with RoPE")


# ── Transformer Block ──────────────────────────────────────────────────


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,  # (batch, seq_len, d_model)
) -> Tensor:
    """A single pre-norm Transformer block with RoPE.

    Architecture:
        x = x + MHA_with_RoPE(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Weights dict keys:
        attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
        attn.output_proj.weight, ln1.weight, ln2.weight,
        ffn.w1.weight, ffn.w2.weight, ffn.w3.weight

    Args:
        d_model: hidden dimension
        num_heads: attention heads
        d_ff: FFN hidden dimension
        max_seq_len: max sequence length
        theta: RoPE base frequency
        weights: state dict with the keys listed above
        in_features: input (batch, seq_len, d_model)

    Returns:
        output (batch, seq_len, d_model)
    """
    raise NotImplementedError("TODO: implement transformer block")


# ── Full Transformer LM ────────────────────────────────────────────────


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,  # (batch, seq_len)
) -> Tensor:
    """Full Transformer language model forward pass.

    Architecture:
        embed = token_embedding(input_ids)
        for each layer: embed = transformer_block(embed)
        embed = RMSNorm(embed)
        logits = lm_head(embed)

    Weights dict keys:
        token_embeddings.weight, lm_head.weight, ln_final.weight,
        layers.{i}.attn.q_proj.weight, layers.{i}.attn.k_proj.weight,
        layers.{i}.attn.v_proj.weight, layers.{i}.attn.output_proj.weight,
        layers.{i}.ln1.weight, layers.{i}.ln2.weight,
        layers.{i}.ffn.w1.weight, layers.{i}.ffn.w2.weight,
        layers.{i}.ffn.w3.weight    (for i in 0..num_layers-1)

    Args:
        vocab_size: vocabulary size
        context_length: max sequence length
        d_model: hidden dimension
        num_layers: number of transformer blocks
        num_heads: attention heads per block
        d_ff: FFN hidden dimension
        rope_theta: RoPE base frequency
        weights: model state dict
        in_indices: input token IDs (batch, seq_len)

    Returns:
        logits (batch, seq_len, vocab_size)
    """
    raise NotImplementedError("TODO: implement full transformer LM forward pass")

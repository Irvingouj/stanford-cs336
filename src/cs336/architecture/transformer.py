"""
Transformer architecture components — from scratch.

Implements the interfaces expected by CS336 Assignment 1:
- RMSNorm, SiLU, SwiGLU
- Scaled dot-product attention
- Multi-head self-attention (with and without RoPE)
- Transformer block
- Transformer LM (full model)

Reference:
    https://github.com/stanford-cs336/assignment1-basics
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Normalization layers ────────────────────────────────────────────────


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
    # RMS = sqrt(mean(x^2))
    rms = torch.sqrt(torch.mean(in_features.float() ** 2, dim=-1, keepdim=True) + eps)
    return (in_features / rms) * weights


class RMSNorm(nn.Module):
    """RMSNorm as a nn.Module (used in LLaMA)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return run_rmsnorm(x.shape[-1], self.eps, self.weight, x)


# ── Activation functions ────────────────────────────────────────────────


def run_silu(in_features: Tensor) -> Tensor:
    """SiLU (Sigmoid Linear Unit) = x * sigmoid(x)."""
    return in_features * torch.sigmoid(in_features)


# ── Linear layer ────────────────────────────────────────────────────────


def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,  # (d_out, d_in)
    in_features: Tensor,  # (..., d_in)
) -> Tensor:
    """Apply a linear transformation.

    Args:
        d_in: input feature dimension
        d_out: output feature dimension
        weights: weight matrix (d_out, d_in)
        in_features: input (..., d_in)

    Returns:
        output (..., d_out)
    """
    return F.linear(in_features, weights)


# ── Embedding ───────────────────────────────────────────────────────────


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,  # (vocab_size, d_model)
    token_ids: Tensor,  # (...)
) -> Tensor:
    """Look up token embeddings.

    Args:
        vocab_size: number of embeddings
        d_model: embedding dimension
        weights: embedding matrix (vocab_size, d_model)
        token_ids: token indices (...)

    Returns:
        embeddings (..., d_model)
    """
    return F.embedding(token_ids, weights)


# ── SwiGLU feed-forward ─────────────────────────────────────────────────


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
        d_ff: hidden dimension (typically 8/3 * d_model rounded to multiple of 256)
        w1_weight, w2_weight, w3_weight: SwiGLU weight matrices
        in_features: input (..., d_model)

    Returns:
        output (..., d_model)
    """
    gate = run_silu(F.linear(in_features, w1_weight))  # (..., d_ff)
    up = F.linear(in_features, w3_weight)               # (..., d_ff)
    return F.linear(gate * up, w2_weight)               # (..., d_model)


class SwiGLU(nn.Module):
    """SwiGLU as a nn.Module."""
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            # LLaMA convention: 8/3 * d_model, rounded to multiple of 256
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return run_swiglu(x.shape[-1], self.w1.out_features,
                          self.w1.weight, self.w2.weight, self.w3.weight, x)


# ── Scaled dot-product attention ────────────────────────────────────────


def run_scaled_dot_product_attention(
    Q: Tensor,  # (..., queries, d_k)
    K: Tensor,  # (..., keys, d_k)
    V: Tensor,  # (..., keys, d_v)
    mask: Tensor | None = None,  # (..., queries, keys)
) -> Tensor:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

    Args:
        Q: queries
        K: keys
        V: values
        mask: optional attention mask (True = masked position)

    Returns:
        attention output (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    scale = d_k ** -0.5

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (..., queries, keys)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


# ── RoPE (Rotary Position Embedding) ────────────────────────────────────


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,  # (..., seq_len, d_k)
    token_positions: Tensor,  # (..., seq_len)
) -> Tensor:
    """Apply Rotary Position Embedding.

    RoPE rotates pairs of dimensions by position-dependent angles.
    angle_i = position * theta^(-2i/d_k)

    Args:
        d_k: head dimension
        theta: base frequency (e.g. 10000.0)
        max_seq_len: max sequence length for pre-computation
        in_query_or_key: Q or K tensor (..., seq_len, d_k)
        token_positions: position indices (..., seq_len)

    Returns:
        RoPE-transformed tensor (..., seq_len, d_k)
    """
    # Compute frequencies: (d_k // 2,)
    i = torch.arange(0, d_k // 2, device=in_query_or_key.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2 * i / d_k))  # (d_k // 2,)

    # Position-dependent angles: (..., seq_len, d_k // 2)
    angles = token_positions.unsqueeze(-1).float() * freqs  # (..., seq_len, d_k // 2)

    cos = torch.cos(angles).unsqueeze(-1)  # (..., seq_len, d_k // 2, 1)
    sin = torch.sin(angles).unsqueeze(-1)

    # Split input into pairs
    x_2d = in_query_or_key.float().view(*in_query_or_key.shape[:-1], d_k // 2, 2)
    x1, x2 = x_2d[..., 0], x_2d[..., 1]  # (..., seq_len, d_k // 2)

    # Rotate: (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated_1 = x1 * cos.squeeze(-1) - x2 * sin.squeeze(-1)
    rotated_2 = x1 * sin.squeeze(-1) + x2 * cos.squeeze(-1)

    result = torch.stack([rotated_1, rotated_2], dim=-1)
    return result.view_as(in_query_or_key).type_as(in_query_or_key)


# ── Multi-head self-attention ───────────────────────────────────────────


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,  # (d_model, d_model)
    k_proj_weight: Tensor,  # (d_model, d_model)
    v_proj_weight: Tensor,  # (d_model, d_model)
    o_proj_weight: Tensor,  # (d_model, d_model)
    in_features: Tensor,  # (..., seq_len, d_model)
) -> Tensor:
    """Multi-head self-attention (without RoPE).

    Args:
        d_model: model dimension
        num_heads: number of attention heads
        q_proj_weight, k_proj_weight, v_proj_weight: Q/K/V projection weights
        o_proj_weight: output projection weights
        in_features: input (..., seq_len, d_model)

    Returns:
        attention output (..., seq_len, d_model)
    """
    *batch_dims, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads

    # Project Q, K, V in one go
    Q = F.linear(in_features, q_proj_weight)  # (..., seq_len, d_model)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)

    # Reshape to (..., num_heads, seq_len, head_dim)
    Q = Q.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)
    K = K.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)
    V = V.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool),
        diagonal=1,
    )

    # Scaled dot-product attention
    attn_out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    # (..., num_heads, seq_len, head_dim)

    # Reassemble heads
    attn_out = attn_out.transpose(-2, -3).contiguous()
    attn_out = attn_out.view(*batch_dims, seq_len, d_model)

    # Output projection
    return F.linear(attn_out, o_proj_weight)


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
    """Multi-head self-attention with RoPE.

    Same as run_multihead_self_attention but applies RoPE to Q and K before attention.
    """
    *batch_dims, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads

    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
        token_positions = token_positions.expand(*batch_dims, seq_len)

    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)

    Q = Q.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)
    K = K.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)
    V = V.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-2, -3)

    # Apply RoPE to Q and K
    Q = run_rope(head_dim, theta, max_seq_len, Q, token_positions)
    K = run_rope(head_dim, theta, max_seq_len, K, token_positions)

    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool),
        diagonal=1,
    )

    attn_out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    attn_out = attn_out.transpose(-2, -3).contiguous()
    attn_out = attn_out.view(*batch_dims, seq_len, d_model)

    return F.linear(attn_out, o_proj_weight)


# ── Transformer block ───────────────────────────────────────────────────


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

    Weights dict keys (see adapters.py for full docs):
        attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
        attn.output_proj.weight, ln1.weight, ln2.weight,
        ffn.w1.weight, ffn.w2.weight, ffn.w3.weight
    """
    # Self-attention sublayer (pre-norm)
    residual = in_features
    x = run_rmsnorm(d_model, 1e-6, weights["ln1.weight"], in_features)
    x = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        x,
    )
    x = x + residual

    # FFN sublayer (pre-norm)
    residual = x
    x = run_rmsnorm(d_model, 1e-6, weights["ln2.weight"], x)
    x = run_swiglu(
        d_model, d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        x,
    )
    return x + residual


# ── Full Transformer LM ─────────────────────────────────────────────────


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

    Weights dict keys:
        token_embeddings.weight, lm_head.weight, ln_final.weight,
        layers.{i}.attn.q_proj.weight, layers.{i}.attn.k_proj.weight,
        layers.{i}.attn.v_proj.weight, layers.{i}.attn.output_proj.weight,
        layers.{i}.ln1.weight, layers.{i}.ln2.weight,
        layers.{i}.ffn.w1.weight, layers.{i}.ffn.w2.weight,
        layers.{i}.ffn.w3.weight

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
    batch_size, seq_len = in_indices.shape

    # Token embeddings
    x = run_embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)

    # Transformer blocks
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {
            "attn.q_proj.weight": weights[prefix + "attn.q_proj.weight"],
            "attn.k_proj.weight": weights[prefix + "attn.k_proj.weight"],
            "attn.v_proj.weight": weights[prefix + "attn.v_proj.weight"],
            "attn.output_proj.weight": weights[prefix + "attn.output_proj.weight"],
            "ln1.weight": weights[prefix + "ln1.weight"],
            "ln2.weight": weights[prefix + "ln2.weight"],
            "ffn.w1.weight": weights[prefix + "ffn.w1.weight"],
            "ffn.w2.weight": weights[prefix + "ffn.w2.weight"],
            "ffn.w3.weight": weights[prefix + "ffn.w3.weight"],
        }
        x = run_transformer_block(
            d_model, num_heads, d_ff,
            context_length, rope_theta,
            layer_weights, x,
        )

    # Final norm
    x = run_rmsnorm(d_model, 1e-6, weights["ln_final.weight"], x)

    # LM head
    return F.linear(x, weights["lm_head.weight"])

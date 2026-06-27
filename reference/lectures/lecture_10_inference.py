"""
Stanford CS336 — Lecture 10: Inference
========================================
Spring 2026. Instructor: Percy Liang.

Topics:
- Autoregressive decoding: greedy, temperature sampling, top-k, top-p
- KV-cache: caching key/value states to avoid recomputation
- Speculative decoding: draft + verify for faster inference
- Quantization: INT8, INT4 (GPTQ, AWQ) for smaller memory footprint

Reference:
- https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py
"""

import torch
import torch.nn.functional as F


# ── Decoding strategies ─────────────────────────────────────────────────

def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """Select the most likely next token."""
    return logits.argmax(dim=-1)


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample with temperature."""
    if temperature == 0:
        return greedy_decode(logits)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Top-k sampling: only sample from the k most likely tokens."""
    logits = logits / temperature
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)
    logits = torch.where(logits < min_top_k, float("-inf"), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0,
) -> torch.Tensor:
    """Nucleus (top-p) sampling: sample from the smallest set with cum prob >= p."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = sorted_probs.cumsum(dim=-1)
    # Mask tokens beyond cumulative p
    mask = cumsum > p
    mask[:, 0] = False  # Always keep at least one token
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # Sample from filtered distribution
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return sorted_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)


# ── KV-Cache ────────────────────────────────────────────────────────────

class KVCache:
    """
    Cache key and value tensors so we don't recompute them for every new token.

    Without KV-cache: O(n^2) attention cost for generating n tokens
    With KV-cache:    O(n) attention cost (each new token attends to all cached K/V)
    """
    def __init__(self):
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K/V tensors (from the latest token)."""
        self.keys.append(k)
        self.values.append(v)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated K/V for all cached tokens."""
        return torch.cat(self.keys, dim=-2), torch.cat(self.values, dim=-2)

    def reset(self):
        self.keys.clear()
        self.values.clear()


def demo_kv_cache():
    """Demonstrate the memory savings of KV-cache."""
    print("── KV-Cache memory analysis ──\n")

    d_model = 4096
    n_layers = 32
    seq_len = 2048
    batch_size = 1
    bytes_per_elem = 2  # FP16

    # Without cache: store full (N,N) attention matrix
    # Actually, PyTorch doesn't store it explicitly but the math is:
    attn_mem = batch_size * n_layers * seq_len * seq_len * bytes_per_elem

    # With cache: store K and V for each layer, each token
    kv_mem = batch_size * n_layers * 2 * seq_len * d_model * bytes_per_elem

    print(f"  Without KV-cache (attention matrix): {attn_mem / 1e9:.2f} GB")
    print(f"  With KV-cache (K+V tensors):         {kv_mem / 1e9:.2f} GB")
    print(f"  Factor:                               {attn_mem / kv_mem:.0f}×")

    # Generative case: 1 token at a time
    # Without cache: each forward pass re-encodes ALL previous tokens
    # With cache: each forward pass only encodes the NEW token
    print(f"\n  Per-token generation cost (no cache):  O(n^2) — reprocess all tokens")
    print(f"  Per-token generation cost (with cache): O(n)   — only process new token")


# ── Quantization sketch ─────────────────────────────────────────────────

def quantize_tensor(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    Simple symmetric quantization: map FP32 to INTn.

    x_q = round(x / scale) * scale
    scale = max(|x|) / (2^(n_bits-1) - 1)
    """
    max_val = 2 ** (n_bits - 1) - 1
    scale = x.abs().max() / max_val
    x_int = torch.round(x / scale).clamp(-max_val, max_val)
    x_deq = x_int * scale
    return x_deq, scale


def demo_quantization():
    print("\n── Quantization demo ──\n")
    x = torch.randn(4, 4) * 3  # Normal-ish, range ~[-9, 9]

    for bits in [8, 4, 2]:
        x_q, scale = quantize_tensor(x, bits)
        error = (x - x_q).abs().mean().item()
        memory = x.numel() * bits / 8
        print(f"  INT{bits}: mean abs error = {error:.6f}, "
              f"memory = {memory:.0f} bytes (vs {x.numel() * 4} for FP32)")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Decoding strategies ──\n")
    torch.manual_seed(42)
    logits = torch.randn(2, 100)  # (batch, vocab)

    print(f"  Greedy:         {greedy_decode(logits).tolist()}")
    print(f"  Temperature 1.0: {temperature_sample(logits, 1.0).tolist()}")
    print(f"  Temperature 0.1: {temperature_sample(logits, 0.1).tolist()}")
    print(f"  Top-k (k=5):     {top_k_sample(logits, k=5).tolist()}")
    print(f"  Top-p (p=0.9):   {top_p_sample(logits, p=0.9).tolist()}")

    demo_kv_cache()
    demo_quantization()

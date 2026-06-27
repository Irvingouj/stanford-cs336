"""
Stanford CS336 — Lecture 3: Architectures & Hyperparameters
============================================================
Spring 2026. Instructor: Tatsu Hashimoto.

Topics:
- Transformer architecture variants (pre-norm, post-norm)
- Positional encodings: sinusoidal, learned, RoPE
- Hyperparameter design space: depth vs width, attention heads
- GPT-2 / LLaMA architecture walkthrough
"""


# ── GPT-2 style transformer block ──────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Standard LayerNorm (not RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, d_model: int, n_heads: int, max_ctx: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask (upper triangular, -inf)
        mask = torch.triu(torch.ones(max_ctx, max_ctx), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # Compute Q, K, V in one projection, then split
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=-1)  # each (B, T, C)

        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        out = attn_weights @ v  # (B, nh, T, hd)

        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """Standard transformer MLP (4x expansion)."""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        inner_dim = d_model * expansion
        self.fc_in = nn.Linear(d_model, inner_dim, bias=False)
        self.fc_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)  # GPT-2 uses GELU
        return self.fc_out(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (modern / LLaMA style)."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: norm before each sublayer
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Mini(nn.Module):
    """Minimal GPT-2-like model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_ctx: int = 1024,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_ctx, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, device=input_ids.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)  # (B, T, vocab_size)


# ── Hyperparameter design space ─────────────────────────────────────────

def hyperparameter_design():
    """Explore the key trade-offs in transformer design."""
    print("── Hyperparameter design trade-offs ──\n")

    # Common configs across model scales
    configs = {
        "GPT-2 Small":    (768, 12, 12),
        "GPT-2 Medium":   (1024, 16, 24),
        "GPT-2 Large":    (1280, 20, 36),
        "GPT-2 XL":       (1600, 25, 48),
        "LLaMA-7B":       (4096, 32, 32),
        "LLaMA-13B":      (5120, 40, 40),
        "LLaMA-70B":      (8192, 64, 80),
    }

    for name, (d, heads, layers) in configs.items():
        params = 12 * layers * d**2  # rough: 12 d^2 per layer
        flops_per_token = 2 * params  # 2x params per token (fwd)
        print(f"  {name:14s}  d={d:<5}  heads={heads:<3}  layers={layers:<3}  "
              f"params≈{params / 1e9:.1f}B  "
              f"head_dim={d // heads}")

    print("\n  Rules of thumb:")
    print("  - Keep head_dim ≈ 64-128")
    print("  - Depth grows with width (roughly d_model ∝ n_layers)")
    print("  - 12 * d_model^2 params per layer is a good estimate")
    print("  - Pre-norm (LLaMA) is more stable than post-norm (GPT-2)")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hyperparameter_design()

    print("\n── Testing GPT2Mini forward pass ──\n")
    model = GPT2Mini(vocab_size=1000, d_model=128, n_heads=4, n_layers=4)
    x = torch.randint(0, 1000, (2, 64))  # (batch=2, seq=64)
    logits = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")  # (2, 64, 1000)
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")

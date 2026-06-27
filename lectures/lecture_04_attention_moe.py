"""
Stanford CS336 — Lecture 4: Attention Alternatives & Mixture of Experts
========================================================================
Spring 2026. Instructor: Tatsu Hashimoto.

Topics:
- Attention alternatives: linear attention, sliding window, sparse
- Mixture of Experts (MoE): architecture, routing, load balancing
- Efficiency trade-offs: compute vs memory vs quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Placeholder — lecture 4 is a PDF in the official repo.
# This file provides starter code for key concepts.


# ── Sliding window attention ────────────────────────────────────────────

def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Create attention mask for sliding window (local) attention.
    Each token attends only to `window_size` tokens before it.
    Used in Mistral, Longformer.
    """
    mask = torch.ones(seq_len, seq_len)
    for i in range(seq_len):
        left = max(0, i - window_size + 1)
        mask[i, :left] = 0  # mask out tokens too far away
    # Also enforce causality
    mask = torch.triu(mask, diagonal=1)
    return mask.bool()


# ── Mixture of Experts (MoE) ────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Instead of one dense FFN, we have N "experts" (smaller FFNs)
    and a router that selects top-k experts per token.

    Key insight: more total parameters but same FLOPs per token
    (only top-k experts are activated).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        expansion: int = 4,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        # Router: produces logits over experts for each token
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Experts: small FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * expansion, bias=False),
                nn.GELU(),
                nn.Linear(d_model * expansion, d_model, bias=False),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, d_model)
        Returns: (batch, seq, d_model)
        """
        B, T, D = x.shape

        # Router logits
        router_logits = self.router(x)  # (B, T, n_experts)

        # Select top-k experts per token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, top_k)

        # Sparse computation: route each token to its top-k experts
        output = torch.zeros_like(x)
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == expert_idx).any(dim=-1)  # (B, T)
            if not mask.any():
                continue

            # Get the weight for this expert
            weight_idx = (top_k_indices == expert_idx).float().argmax(dim=-1)  # (B, T)
            weights = top_k_weights.gather(-1, weight_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

            # Process tokens
            expert_input = x[mask]  # (n_tokens, D)
            expert_output = self.experts[expert_idx](expert_input)
            output[mask] += expert_output * weights[mask].unsqueeze(-1)

        return output


# ── Load balancing loss ─────────────────────────────────────────────────

def load_balancing_loss(router_logits: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Auxiliary loss to encourage uniform expert usage.
    Without this, the router collapses to using only 1-2 experts.

    router_logits: (B*T, n_experts)
    Returns: scalar loss
    """
    # Fraction of tokens dispatched to each expert
    n_tokens = router_logits.shape[0]
    n_experts = router_logits.shape[1]

    # Expert assignment counts
    expert_counts = torch.bincount(
        top_k_indices.flatten(), minlength=n_experts
    ).float()
    fraction = expert_counts / (n_tokens * top_k_indices.shape[1])

    # Mean router probability per expert
    mean_prob = router_logits.softmax(-1).mean(0)

    # Load balancing loss: encourages uniform distribution
    loss = n_experts * (fraction * mean_prob).sum()
    return loss


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Sliding window mask ──\n")
    mask = sliding_window_mask(seq_len=8, window_size=3)
    print(mask.int())
    print("\nEach row = query, each col = key; 1 = attend, 0 = masked\n")

    print("── MoE forward pass ──\n")
    moe = MoELayer(d_model=256, n_experts=8, top_k=2)
    x = torch.randn(4, 16, 256)  # (batch, seq, d_model)
    y = moe(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"Total params: {total_params:,} ({total_params / 1e6:.1f}M)")
    # Active params per token is roughly 2/top_k * total_params / n_experts
    active_params = total_params * 2 / 8
    print(f"Active params per token: ~{active_params:,} ({active_params / 1e6:.1f}M)")

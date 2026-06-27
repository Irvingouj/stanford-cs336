"""
Stanford CS336 — Lectures 15 & 16: Mid/Post-Training
======================================================
Spring 2026. Instructor: Tatsu Hashimoto.

Topics (L15 — SFT & RLHF):
- Supervised Fine-Tuning (SFT): training on instruction-response pairs
- RLHF: reward model training, PPO, DPO
- Preference data: how to collect and use it

Topics (L16 — RLVR):
- RL with Verifiable Rewards for math/code reasoning
- GRPO (Group Relative Policy Optimization)
- Outcome-based vs process-based rewards

Reference:
- DPO: https://arxiv.org/abs/2305.18290
- RLHF: https://arxiv.org/abs/2203.02155
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Supervised Fine-Tuning (SFT) ────────────────────────────────────────

def sft_loss(
    logits: torch.Tensor,  # (batch, seq, vocab)
    labels: torch.Tensor,  # (batch, seq)
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Standard next-token prediction loss, used for both pretraining and SFT.

    In SFT, we typically mask out the prompt tokens and only compute loss
    on the response tokens:

        prompt: "<|user|> What is 2+2? <|assistant|>"
        labels: [-100, -100, -100, -100, -100, -100, 4 ("4")]
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


# ── Direct Preference Optimization (DPO) ────────────────────────────────

def dpo_loss(
    policy_logps_chosen: torch.Tensor,  # log π_θ(y_w | x)  per sample
    policy_logps_rejected: torch.Tensor,  # log π_θ(y_l | x)
    ref_logps_chosen: torch.Tensor,  # log π_ref(y_w | x)
    ref_logps_rejected: torch.Tensor,  # log π_ref(y_l | x)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss (Rafailov et al., 2023).

    L_DPO = -E[ log σ( β [ log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l) ] ) ]

    Intuition: increase the relative probability of chosen over rejected,
    compared to the reference model. The beta parameter controls how far
    the policy can deviate from the reference.
    """
    policy_ratio = policy_logps_chosen - policy_logps_rejected
    ref_ratio = ref_logps_chosen - ref_logps_rejected
    logits = policy_ratio - ref_ratio
    return -F.logsigmoid(beta * logits).mean()


# ── Group Relative Policy Optimization (GRPO) ────────────────────────────

def grpo_loss(
    rewards: torch.Tensor,  # (batch, n_samples_per_prompt)
    logprobs: torch.Tensor,  # (batch, n_samples_per_prompt)
    old_logprobs: torch.Tensor,  # (batch, n_samples_per_prompt)
    clip_eps: float = 0.2,
    kl_penalty: float = 0.01,
) -> torch.Tensor:
    """
    GRPO-style loss for RLVR (used in DeepSeek-R1 training).

    Key ideas:
    1. Group-relative advantage: normalize rewards within each prompt's samples
    2. Clipped importance sampling (like PPO)
    3. KL penalty from reference (optional, often omitted in GRPO)
    """
    # Group-relative advantage: (r - mean) / std within each prompt
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantage = advantage.unsqueeze(-1)

    # Importance sampling ratio
    ratio = torch.exp(logprobs - old_logprobs)

    # Clipped objective
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    # Optional KL penalty
    kl = (old_logprobs - logprobs).mean()
    return policy_loss + kl_penalty * kl


# ── Reward model training ───────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Bradley-Terry reward model for RLHF.

    Given a prompt x and response y, predict a scalar reward r(x, y).
    Trained on preference pairs: P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq, d_model)
        pooled = hidden_states.mean(dim=1)  # mean pooling
        return self.backbone(pooled)  # (batch, 1)

    @staticmethod
    def reward_loss(
        r_chosen: torch.Tensor,
        r_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """Bradley-Terry loss for preference pairs."""
        return -F.logsigmoid(r_chosen - r_rejected).mean()


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── SFT Loss ──\n")
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 1000)
    labels = torch.randint(0, 1000, (2, 10))
    labels[:, :5] = -100  # mask prompt tokens
    loss = sft_loss(logits, labels)
    print(f"  SFT loss (masked prompt): {loss.item():.4f}")

    print("\n── DPO Loss ──\n")
    torch.manual_seed(42)
    n = 8
    # Simulate: model prefers chosen over rejected
    policy_chosen = torch.randn(n) * 0.5 - 1.0  # slightly higher
    policy_rejected = torch.randn(n) * 0.5 - 2.0  # lower
    ref_chosen = torch.randn(n) * 0.5 - 1.5
    ref_rejected = torch.randn(n) * 0.5 - 1.5

    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    print(f"  DPO loss: {loss.item():.4f}")
    print(f"  Mean logp chosen:  {policy_chosen.mean():.2f}")
    print(f"  Mean logp rejected: {policy_rejected.mean():.2f}")

    print("\n── GRPO Loss ──\n")
    # Simulate: 4 prompts, 8 samples each
    rewards = torch.randn(32) + 1.0
    rewards[::2] += 1.0  # every other is better
    logprobs = torch.randn(32) * 0.3
    old_logprobs = logprobs + torch.randn(32) * 0.1

    loss_grpo = grpo_loss(rewards, logprobs, old_logprobs)
    print(f"  GRPO loss: {loss_grpo.item():.4f}")

    print("\n── Reward Model ──\n")
    rm = RewardModel(d_model=64)
    h_chosen = torch.randn(4, 16, 64)   # chosen responses
    h_rejected = torch.randn(4, 16, 64)  # rejected responses
    r_chosen = rm(h_chosen)
    r_rejected = rm(h_rejected)
    r_loss = RewardModel.reward_loss(r_chosen, r_rejected)
    print(f"  Reward model loss: {r_loss.item():.4f}")
    print(f"  Mean reward chosen:  {r_chosen.mean().item():.2f}")
    print(f"  Mean reward rejected: {r_rejected.mean().item():.2f}")

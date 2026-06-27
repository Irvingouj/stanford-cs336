"""
Stanford CS336 — Lecture 17: Alignment & Multimodality
========================================================
Spring 2026. Instructor: Percy Liang.

Topics:
- Alignment tax: does making models "safer" reduce capability?
- Constitutional AI: self-critique + self-revision
- Red-teaming: adversarial testing for safety
- Multimodality: vision-language models, image generation
- The frontier: GPT-4V, Gemini, Claude 3.5 Vision

This is a conceptual/PDF lecture. Code provided for alignment experiments.
"""

import torch
import torch.nn.functional as F


# ── Constitutional AI sketch ────────────────────────────────────────────

def constitutional_ai_pipeline():
    """
    Anthropic's Constitutional AI (simplified):

    Phase 1 — Supervised:
    1. Generate harmful responses from base model
    2. Ask model to critique its own response according to a "constitution"
    3. Ask model to revise based on critique
    4. Fine-tune on revised responses

    Phase 2 — RL:
    1. Train a "harmlessness" reward model using AI-generated preferences
    2. RL fine-tuning with helpfulness + harmlessness reward

    The "constitution" is a list of principles, e.g.:
    - "Choose the response that is least harmful."
    - "Don't give advice about committing crimes."
    - "Respect human autonomy and dignity."
    """
    principles = [
        "Choose the response that is least harmful to humans.",
        "Do not assist with illegal activities.",
        "Respect privacy and confidentiality.",
        "Be honest about your limitations and uncertainties.",
        "Avoid stereotyping or discrimination.",
    ]
    print("── Constitutional AI Principles ──\n")
    for i, p in enumerate(principles, 1):
        print(f"  {i}. {p}")


# ── Safety vs helpfulness trade-off ─────────────────────────────────────

def alignment_tax_demo():
    """
    The "alignment tax": RLHF can reduce capabilities.

    Hypothetical data showing the trade-off:
    """
    print("\n── Alignment Tax (hypothetical) ──\n")

    models = [
        ("Base pretrained",  85.0, 40.0),
        ("SFT only",         84.5, 55.0),
        ("RLHF (helpful)",   83.0, 82.0),
        ("RLHF (safe)",      78.0, 95.0),
        ("RLHF (balanced)",  80.5, 90.0),
    ]

    print(f"  {'Model':<20s} {'MMLU %':>8s}  {'Safety %':>8s}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}")
    for name, mmlu, safety in models:
        print(f"  {name:<20s} {mmlu:>7.1f}  {safety:>8.1f}")
    print()
    print("  Trade-off: safer models sometimes score lower on benchmarks.")
    print("  Current research goal: minimize or eliminate this gap.")


# ── DPO for safety alignment ────────────────────────────────────────────

def preference_pair_example():
    """
    Example preference pair for safety DPO:

    Prompt: "How do I make napalm?"

    Chosen (safe):
      "I can't provide instructions for making napalm or other
       dangerous substances. If you're interested in chemistry,
       I'd be happy to discuss safe experiments instead."

    Rejected (unsafe):
      "Here's how to make napalm: mix styrofoam with gasoline..."

    The DPO loss (see lecture_15_16) directly optimizes to prefer
    the safe response over the unsafe one.
    """
    print("\n── Safety preference pair (DPO) ──\n")
    print(preference_pair_example.__doc__)


# ── Multimodality overview ──────────────────────────────────────────────

def multimodality_overview():
    """Key concepts in vision-language models."""
    print("── Multimodality: Vision-Language Models ──\n")

    approaches = {
        "CLIP-style": "Separate text + image encoders, cosine similarity.",
        "Flamingo-style": "Visual tokens interleaved with text in a pretrained LM.",
        "LLaVA-style": "Vision encoder → projection layer → LLM.",
        "GPT-4V/Gemini": "Proprietary; likely similar to LLaVA at scale.",
        "Diffusion (DALL-E/SD)": "Text conditioning for image generation, separate paradigm.",
    }

    for name, desc in approaches.items():
        print(f"  {name:20s}  {desc}")

    print("\n  Key challenges:")
    print("  - Aligning visual and textual representations")
    print("  - Computational cost (high-res images → many tokens)")
    print("  - Evaluation: harder than text-only benchmarks")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    constitutional_ai_pipeline()
    alignment_tax_demo()
    preference_pair_example()
    multimodality_overview()

"""
Stanford CS336 — Lecture 12: Evaluation
=========================================
Spring 2026. Instructor: Percy Liang.

Topics:
- Perplexity: the fundamental LM metric
- Benchmark suites: MMLU, HellaSwag, ARC, GSM8K, HumanEval, BBH
- LM Evaluation Harness (EleutherAI)
- Contamination detection
- The pitfalls of evaluation: data leakage, prompt sensitivity

Reference:
- https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py
- https://github.com/EleutherAI/lm-evaluation-harness
"""

import torch
import torch.nn.functional as F
import math


# ── Perplexity ──────────────────────────────────────────────────────────

def compute_perplexity(
    logits: torch.Tensor,  # (batch, seq, vocab)
    target_ids: torch.Tensor,  # (batch, seq)
    ignore_index: int = -100,
) -> float:
    """
    Perplexity = exp(cross-entropy loss).

    Intuition: if perplexity = 10, the model is as confused as if it
    were choosing uniformly among 10 options at each step.
    """
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = target_ids.view(-1)

    # Filter padding
    mask = targets_flat != ignore_index
    logits_flat = logits_flat[mask]
    targets_flat = targets_flat[mask]

    loss = F.cross_entropy(logits_flat, targets_flat)
    return math.exp(loss.item())


# ── Benchmark overview ──────────────────────────────────────────────────

BENCHMARKS = {
    "MMLU": {
        "description": "Massive Multitask Language Understanding",
        "domains": "57 subjects: STEM, humanities, social sciences, etc.",
        "format": "Multiple choice (A/B/C/D)",
        "metric": "Accuracy",
        "example": "What is the capital of France? A) London B) Paris C) Berlin D) Madrid",
    },
    "HellaSwag": {
        "description": "Commonsense NLI: pick the most plausible ending",
        "domains": "Everyday scenarios, activitynet, wikiHow",
        "format": "4-way multiple choice",
        "metric": "Accuracy",
    },
    "ARC": {
        "description": "AI2 Reasoning Challenge: grade-school science questions",
        "domains": "Science (Easy + Challenge sets)",
        "format": "Multiple choice",
        "metric": "Accuracy",
    },
    "GSM8K": {
        "description": "Grade School Math: multi-step math word problems",
        "domains": "Arithmetic, word problems",
        "format": "Free-form answer (number)",
        "metric": "Exact match",
    },
    "HumanEval": {
        "description": "Code generation: write Python functions from docstrings",
        "domains": "Programming",
        "format": "Code completion",
        "metric": "pass@k",
    },
    "BBH": {
        "description": "BIG-Bench Hard: 23 challenging tasks from BIG-Bench",
        "domains": "Reasoning, math, logic, etc.",
        "format": "Varies by task",
        "metric": "Accuracy / exact match",
    },
}


def benchmark_overview():
    """Display the major LM evaluation benchmarks."""
    print("── Major LM benchmarks ──\n")
    for name, info in BENCHMARKS.items():
        print(f"  {name}:")
        print(f"    {info['description']}")
        print(f"    Format: {info['format']} | Metric: {info['metric']}")
        print()


# ── Contamination detection ─────────────────────────────────────────────

def ngram_overlap(
    train_text: str,
    eval_text: str,
    n: int = 13,
) -> float:
    """
    Simple n-gram overlap for contamination detection.
    If an eval example appears verbatim in training data, it's contaminated.
    """
    def get_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    train_ngrams = get_ngrams(train_text, n)
    eval_ngrams = get_ngrams(eval_text, n)

    if not eval_ngrams:
        return 0.0

    overlap = len(train_ngrams & eval_ngrams)
    return overlap / len(eval_ngrams)


# ── Prompt sensitivity ──────────────────────────────────────────────────

def prompt_sensitivity_demo():
    """Show how different prompts give different results for the same question."""
    print("── Prompt sensitivity ──\n")
    print("Same question, different prompts → different model behavior:\n")

    prompts = [
        "Q: What is 2+2?\nA:",
        "Solve: 2+2 =",
        "Let's think step by step.\nWhat is 2+2?",
        "You are a math expert. Calculate: 2+2 = ?",
    ]

    for p in prompts:
        print(f"  {p!r}")
    print()
    print("  This is why benchmark results depend heavily on prompting format.")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Perplexity demo
    print("── Perplexity ──\n")
    torch.manual_seed(42)
    vocab_size = 1000

    # Perfect model: all probability mass on correct token
    logits = torch.full((2, 10, vocab_size), -1e9)
    targets = torch.randint(0, vocab_size, (2, 10))
    for b in range(2):
        for s in range(10):
            logits[b, s, targets[b, s]] = 0.0

    ppl = compute_perplexity(logits, targets)
    print(f"  Perplexity (near-perfect): {ppl:.2f}  (should be ≈ 1)")

    # Random model
    logits_random = torch.randn(2, 10, vocab_size)
    ppl_random = compute_perplexity(logits_random, targets)
    print(f"  Perplexity (random):      {ppl_random:.2f}  (should be ≈ {vocab_size})")
    print()

    benchmark_overview()

    # Contamination demo
    train = "the quick brown fox jumps over the lazy dog"
    eval_clean = "stanford cs336 language modeling from scratch"
    eval_contam = "the quick brown fox jumps over the fence"
    print("── Contamination detection (13-gram overlap) ──\n")
    print(f"  Clean example:      {ngram_overlap(train, eval_clean):.0%}")
    print(f"  Contaminated:       {ngram_overlap(train, eval_contam):.0%}")
    print()

    prompt_sensitivity_demo()

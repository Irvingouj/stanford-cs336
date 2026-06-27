"""
Stanford CS336 — Lectures 13 & 14: Data
=========================================
Spring 2026. Instructor: Percy Liang.

Topics (L13 — Sources & Datasets):
- Data sources: Common Crawl, Wikipedia, books, code, scientific papers
- Dataset compositions: The Pile, RefinedWeb, FineWeb, Dolma
- Web crawling basics

Topics (L14 — Filtering, Dedup, Mixing, Synthetic):
- Quality filtering: perplexity-based, classifier-based, heuristic
- Deduplication: exact, fuzzy (MinHash, SimHash), URL-based
- Data mixing: domain weighting, annealing
- Synthetic data generation

Reference:
- https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py
- https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py
"""

import hashlib
from collections import Counter


# ── MinHash deduplication ───────────────────────────────────────────────

class MinHash:
    """
    MinHash for approximate near-duplicate detection.

    The idea: hash each n-gram with k different hash functions;
    the minimum hash value for each function forms the signature.
    Two documents with similar Jaccard similarity will have similar signatures.
    """
    def __init__(self, n_hashes: int = 128, seed: int = 42):
        self.n_hashes = n_hashes
        self.seed = seed

    def signature(self, tokens: list[str], n: int = 5) -> list[int]:
        """Compute MinHash signature for a document."""
        ngrams = {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
        if not ngrams:
            return [0] * self.n_hashes

        sig = []
        for i in range(self.n_hashes):
            min_hash = float("inf")
            for ngram in ngrams:
                h = self._hash(ngram, i)
                min_hash = min(min_hash, h)
            sig.append(min_hash)
        return sig

    def _hash(self, text: str, idx: int) -> int:
        h = hashlib.sha256(f"{text}{idx}{self.seed}".encode()).hexdigest()
        return int(h[:16], 16)

    @staticmethod
    def similarity(sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity between two signatures."""
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


# ── Quality filtering ───────────────────────────────────────────────────

def heuristic_filter(text: str) -> tuple[bool, str]:
    """
    Apply heuristic filters to decide if text is high-quality.

    Common filters used in practice (Dolma, FineWeb):
    - Length: reject very short documents
    - Word-to-char ratio: reject gibberish
    - Mean word length: reject if too long (e.g., base64, code dumps)
    - Special character ratio: reject if too high
    - Perplexity score: optional (needs a model)
    """
    if len(text) < 100:
        return False, "too_short"

    words = text.split()
    if len(words) == 0:
        return False, "no_words"

    char_count = len(text)
    word_count = len(words)

    # Word-to-char ratio: typical English is ~0.15-0.25
    wc_ratio = word_count / char_count if char_count > 0 else 0
    if wc_ratio < 0.05:
        return False, f"low_word_char_ratio ({wc_ratio:.3f})"

    # Mean word length
    mean_wl = sum(len(w) for w in words) / word_count
    if mean_wl > 20:
        return False, f"high_mean_word_length ({mean_wl:.1f})"

    # Special character ratio
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / char_count
    if special_ratio > 0.5:
        return False, f"high_special_char_ratio ({special_ratio:.3f})"

    return True, "pass"


# ── Data mixing ─────────────────────────────────────────────────────────

def data_mixing_weights(domains: dict[str, int], target_weights: dict[str, float]):
    """
    Given domain sizes and target sampling weights, compute per-sample
    probability for each domain.

    domains = {"wikipedia": 1_000_000, "books": 500_000, "code": 300_000}
    target_weights = {"wikipedia": 0.3, "books": 0.4, "code": 0.3}
    """
    total = sum(domains.values())
    probs = {}
    for domain, size in domains.items():
        target = target_weights.get(domain, 0)
        # P(sample from domain) = target_weight / fraction_of_data
        probs[domain] = target / (size / total)
    # Normalize
    norm = sum(probs.values())
    return {k: v / norm for k, v in probs.items()}


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Heuristic quality filtering ──\n")

    examples = [
        "This is a proper English sentence about language models and their training data pipelines.",
        "asdf jkl; qwerty zxcvbnm 12345 @@@ ### !!!",
        "This sentence is too short.",
    ]

    for text in examples:
        ok, reason = heuristic_filter(text)
        status = "PASS" if ok else f"REJECT ({reason})"
        print(f"  {status}: {text!r}")

    print("\n── MinHash near-dedup ──\n")
    mh = MinHash(n_hashes=64)

    doc1 = "the quick brown fox jumps over the lazy dog".split()
    doc2 = "the quick brown fox jumps over the lazy cat".split()
    doc3 = "stanford cs336 language modeling from scratch".split()

    s12 = MinHash.similarity(mh.signature(doc1), mh.signature(doc2))
    s13 = MinHash.similarity(mh.signature(doc1), mh.signature(doc3))
    print(f"  Doc1 vs Doc2 (similar): {s12:.3f}")
    print(f"  Doc1 vs Doc3 (different): {s13:.3f}")

    print("\n── Data mixing ──\n")
    domains = {"wikipedia": 1_000_000, "books": 500_000, "code": 300_000, "web": 5_000_000}
    targets = {"wikipedia": 0.2, "books": 0.3, "code": 0.3, "web": 0.2}
    probs = data_mixing_weights(domains, targets)
    for domain, p in probs.items():
        print(f"  {domain:12s}: {p:.3f}")
    print(f"\n  Sum of weights across domains for uniform sampling: {sum(probs.values()):.3f}")

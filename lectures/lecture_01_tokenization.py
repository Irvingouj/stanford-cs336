"""
Stanford CS336 — Lecture 1: Overview & Tokenization
====================================================
Spring 2026. Instructor: Percy Liang.

What is a language model?
A probability distribution over sequences of tokens:
    P(w_1, ..., w_n) = Π_i P(w_i | w_1, ..., w_{i-1})

Tokenization is the first step: breaking raw text into discrete tokens
that the model can process.

Topics:
- BPE (Byte-Pair Encoding) — GPT-2/3/4
- WordPiece — BERT
- SentencePiece / Unigram — LLaMA, T5
- tiktoken — OpenAI's fast BPE

References:
- https://stanford-cs336.github.io
- https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
"""

import tiktoken


# ── Explore OpenAI's tokenizer ──────────────────────────────────────────

def explore_tiktoken():
    """Play with tiktoken to understand tokenization behavior."""
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    examples = [
        "Stanford CS336: Language Modeling from Scratch",
        "Hello world!",
        "The transformer architecture revolutionized NLP.",
        "Tokenization is tricky: don't, can't, it's.",
        "中文：语言模型从零开始",  # Chinese: LM from scratch
    ]

    for text in examples:
        tokens = enc.encode(text)
        pieces = [enc.decode_single_token_bytes(t) for t in tokens]
        print(f"  text:   {text!r}")
        print(f"  tokens: {tokens}")
        print(f"  pieces: {pieces}")
        print()

    print(f"Vocabulary size: {enc.n_vocab:,}")


# ── Build a minimal BPE tokenizer ───────────────────────────────────────

def build_bpe_tokenizer():
    """
    Implement BPE from scratch.
    Algorithm:
    1. Start with each byte as a token (256 base tokens)
    2. Count adjacent token pairs
    3. Merge the most frequent pair into a new token
    4. Repeat until desired vocabulary size
    """
    from collections import Counter

    def get_stats(ids: list[int]) -> Counter:
        """Count frequencies of adjacent token pairs."""
        return Counter(zip(ids, ids[1:]))

    def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` with `new_id`."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    # Small corpus for demonstration
    corpus = "aaabdaaabac"
    print(f"Corpus: {corpus!r}")
    print(f"Unique chars: {sorted(set(corpus))}")

    # Map chars to initial token ids
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}  # string -> id
    itos = {i: ch for i, ch in enumerate(chars)}  # id -> string

    ids = [stoi[ch] for ch in corpus]
    print(f"Initial ids: {ids}")
    print(f"Initial vocab size: {len(chars)}")

    # Run BPE for a few merges
    vocab_size = len(chars)
    merges: dict[tuple[int, int], int] = {}
    num_merges = 5

    for step in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        top_pair = stats.most_common(1)[0][0]
        new_id = vocab_size
        print(f"  Merge {step + 1}: {top_pair} -> {new_id} "
              f"(freq: {stats[top_pair]})")

        ids = merge(ids, top_pair, new_id)
        merges[top_pair] = new_id
        vocab_size += 1

    # Show final token sequence
    def decode(ids: list[int]) -> str:
        """Naive decode back to string."""
        result = []
        for tid in ids:
            if tid in itos:
                result.append(itos[tid])
            else:
                # A merged token: reconstruct it
                result.append(f"<{tid}>")
        return "".join(result)

    print(f"\nFinal ids: {ids}")
    print(f"Decoded:  {decode(ids)}")
    print(f"Final vocab size: {vocab_size}")
    print(f"Merges performed: {merges}")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Exploring tiktoken (GPT-4 tokenizer)")
    print("=" * 60)
    explore_tiktoken()

    print("=" * 60)
    print("Building BPE from scratch")
    print("=" * 60)
    build_bpe_tokenizer()

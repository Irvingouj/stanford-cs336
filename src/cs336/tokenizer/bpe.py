"""
BPE Tokenizer — from scratch implementation.

Implements the interfaces expected by CS336 Assignment 1:
- get_tokenizer(): construct a BPE tokenizer from vocab + merges
- run_train_bpe(): train BPE on a corpus

Reference:
    https://github.com/stanford-cs336/assignment1-basics
"""

from __future__ import annotations

import os
from collections.abc import Iterable


# ── BPE Tokenizer class ─────────────────────────────────────────────────


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    Attributes:
        vocab: mapping from token ID to token bytes
        merges: ordered list of merge pairs
        special_tokens: tokens that are never split
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build reverse index: bytes → id
        self._bytes_to_id = {v: k for k, v in vocab.items()}

        # Build merge priority: pair → rank
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        """Encode a string into token IDs."""
        raise NotImplementedError("TODO: implement BPE encoding")

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs back to a string."""
        raise NotImplementedError("TODO: implement BPE decoding")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ── Assignment interface functions ──────────────────────────────────────


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """Construct a BPE tokenizer from a vocabulary and merge list.

    Args:
        vocab: mapping int → bytes (token ID → token bytes)
        merges: ordered list of (token1_bytes, token2_bytes) merges
        special_tokens: tokens that are never split

    Returns:
        BPETokenizer instance
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a corpus.

    Args:
        input_path: path to training text file
        vocab_size: desired vocabulary size (including special tokens)
        special_tokens: tokens that should never be split

    Returns:
        (vocab, merges) tuple — vocab maps int→bytes, merges is ordered
    """
    raise NotImplementedError("TODO: implement BPE training from scratch")

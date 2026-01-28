"""Utility functions for BPE tokenizer training and serialization."""

import json
from pathlib import Path


def serialize_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """
    Serialize vocabulary and merges to disk.

    Args:
        vocab: The tokenizer vocabulary mapping token IDs to bytes
        merges: List of BPE merges (token1, token2) in merge order
        vocab_path: Path to save vocabulary JSON file
        merges_path: Path to save merges JSON file
    """
    vocab_json = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    merges_json = [[token1.decode("latin-1"), token2.decode("latin-1")] for token1, token2 in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes, int]:
    """
    Find the longest token in the vocabulary.

    Args:
        vocab: The tokenizer vocabulary mapping token IDs to bytes

    Returns:
        Tuple of (token_id, token_bytes, token_length)
    """
    longest_id = -1
    longest_token = b""
    longest_len = 0

    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > longest_len:
            longest_id = token_id
            longest_token = token_bytes
            longest_len = len(token_bytes)

    return longest_id, longest_token, longest_len


def train_and_save(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer and save vocabulary and merges to disk.

    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
        vocab_path: Path to save vocabulary JSON file
        merges_path: Path to save merges JSON file

    Returns:
        Tuple of (vocab, merges)
    """
    from cs336_basics.tokenizer.bpe import train_bpe

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    serialize_vocab_and_merges(vocab, merges, vocab_path, merges_path)

    return vocab, merges

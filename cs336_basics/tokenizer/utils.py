"""Utility functions for BPE tokenizer training and serialization."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (0-255) and a printable unicode character.
    This matches the GPT-2 byte encoder scheme to avoid invisible/control characters in JSON.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


@lru_cache
def unicode_to_bytes() -> dict[str, int]:
    """Inverse of bytes_to_unicode for decoding GPT-2 style strings."""
    return {v: k for k, v in bytes_to_unicode().items()}


def encode_token_gpt2(token_bytes: bytes) -> str:
    """Encode raw token bytes into printable GPT-2 style unicode string."""
    b2u = bytes_to_unicode()
    return "".join(b2u[b] for b in token_bytes)


def decode_token_gpt2(token_str: str) -> bytes:
    """Decode GPT-2 style unicode string back into raw token bytes."""
    u2b = unicode_to_bytes()
    try:
        return bytes(u2b[ch] for ch in token_str)
    except KeyError as exc:
        raise ValueError(f"Token contains non-GPT2 byte-encoding characters: {token_str!r}") from exc


def _encode_merges_gpt2(merges: Iterable[tuple[bytes, bytes]]) -> list[list[str]]:
    return [[encode_token_gpt2(t1), encode_token_gpt2(t2)] for t1, t2 in merges]


def serialize_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """
    Serialize vocabulary and merges using GPT-2 byte encoder (printable unicode).

    Output format:
    - vocab: token(str) -> id(int), similar to tests/fixtures/gpt2_vocab.json
    - merges: JSON list of [token1, token2], with tokens encoded via bytes_to_unicode
    """
    vocab_json = {encode_token_gpt2(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, separators=(",", ":"))

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, pair in enumerate(_encode_merges_gpt2(merges)):
            line = json.dumps(pair, ensure_ascii=False, separators=(",", ":"))
            if i < len(merges) - 1:
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")
        f.write("]\n")


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

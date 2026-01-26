"""
Train a byte-level BPE tokenizer on the owt dataset.

This script trains a BPE tokenizer with vocab_size=10,000 including the <|endoftext|> special token,
serializes the vocabulary and merges to disk, and profiles the training process.
"""

import json
import time
import tracemalloc
import cProfile
import pstats
from io import StringIO
from pathlib import Path

from cs336_basics.bpe import train_bpe


def serialize_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """
    Serialize vocabulary and merges to disk.

    Args:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of BPE merge operations
        vocab_path: Path to save vocabulary JSON
        merges_path: Path to save merges file
    """
    # Serialize vocabulary as JSON (convert bytes to string for JSON)
    vocab_json = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Serialize merges as text file (one merge per line, space-separated)
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            # Use repr-style encoding to handle non-printable bytes
            token1_str = token1.decode("utf-8", errors="replace")
            token2_str = token2.decode("utf-8", errors="replace")
            f.write(f"{token1_str} {token2_str}\n")


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes, int]:
    """
    Find the longest token in the vocabulary.

    Returns:
        Tuple of (token_id, token_bytes, length)
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


def main():
    # Configuration
    input_path = "/data/assignment1-data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "owt_vocab.json"
    merges_path = output_dir / "owt_merges.txt"

    print("=" * 80)
    print("Training BPE Tokenizer on owt Dataset")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"Output directory: {output_dir}")
    print()

    # Start memory tracking
    tracemalloc.start()

    # Profile the training process
    profiler = cProfile.Profile()
    profiler.enable()

    # Time the training
    start_time = time.time()

    print("Starting BPE training...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    end_time = time.time()
    profiler.disable()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate training time
    training_time = end_time - start_time

    print(f"✓ BPE training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"✓ Peak memory usage: {peak / 1024**3:.2f} GB")
    print()

    # Serialize to disk
    print("Serializing vocabulary and merges to disk...")
    serialize_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    print(f"✓ Vocabulary saved to: {vocab_path}")
    print(f"✓ Merges saved to: {merges_path}")
    print()

    # Find longest token
    longest_id, longest_token, longest_len = find_longest_token(vocab)
    print("=" * 80)
    print("Training Statistics")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes, {training_time/3600:.2f} hours)")
    print(f"Peak memory: {peak / 1024**3:.2f} GB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print()
    print(f"Longest token:")
    print(f"  ID: {longest_id}")
    print(f"  Length: {longest_len} bytes")
    print(f"  Bytes: {longest_token}")
    try:
        decoded = longest_token.decode("utf-8")
        print(f"  Decoded: {repr(decoded)}")
    except UnicodeDecodeError:
        print(f"  Decoded: <unable to decode as UTF-8>")
    print()

    # Print profiling results
    print("=" * 80)
    print("Profiling Results - Top 20 Time-Consuming Functions")
    print("=" * 80)
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(s.getvalue())

    # Save profiling results to file
    profile_path = output_dir / "owt_bpe_training_profile.txt"
    with open(profile_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()
    print(f"Full profiling results saved to: {profile_path}")
    print()

    print("=" * 80)
    print("Summary for Assignment Deliverable")
    print("=" * 80)
    print(f"Training completed in {training_time/60:.2f} minutes using {peak / 1024**3:.2f} GB RAM. " +
          f"The longest token is {longest_len} bytes long: {repr(longest_token.decode('utf-8', errors='replace'))}, " +
          f"which makes sense as it likely represents a common multi-character sequence in the owt dataset.")
    print()


if __name__ == "__main__":
    main()

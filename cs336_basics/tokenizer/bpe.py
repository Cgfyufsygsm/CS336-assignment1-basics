"""
BPE Training using flattened pair list structure.
Key optimization: avoid creating new tuples on every merge by using in-place updates.
"""
import heapq
import os
from collections import defaultdict
from typing import Optional

from cs336_basics.tokenizer.pretokenization import pretokenize
from cs336_basics.utils import get_logger, timer, ProgressBar

BYTE_TOKENS = [bytes([i]) for i in range(256)]

logger = get_logger(__name__)


class _PairEntry:
    """Wrapper for heap entries to implement max-heap with lexicographic tiebreaking."""
    __slots__ = ('count', 'pair')

    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: '_PairEntry') -> bool:
        # Max-heap: higher count comes first
        # Tiebreak: lexicographically larger pair comes first
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


@timer
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer using flattened pair list structure.

    This implementation uses a flat list of pairs with logical deletion,
    avoiding expensive tuple recreation on every merge.
    """
    logger.info(f"Starting BPE training on {input_path}")
    logger.info(f"Target vocab size: {vocab_size}, Special tokens: {special_tokens}")

    # Step 1: Pretokenization
    pretoken_freqs = pretokenize(input_path, special_tokens)
    logger.info(f"Unique pretokens: {len(pretoken_freqs):,}")

    # Step 2: Build vocabulary
    logger.info("Initializing vocabulary and pair structures...")
    vocab: dict[int, bytes] = {}
    next_id = 0
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    for b in range(256):
        vocab[next_id] = BYTE_TOKENS[b]
        next_id += 1

    logger.info(f"Initial vocabulary size: {len(vocab)} (special tokens + 256 bytes)")

    # Flatten all pretokens into a list of pairs
    byte_pairs: list[Optional[tuple[tuple[bytes, bytes], int]]] = []
    byte_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    bp_indices: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)

    logger.info(f"Building pair structures from {len(pretoken_freqs):,} unique pretokens...")
    with ProgressBar() as pbar:
        task_id = pbar.add_task("[yellow]Building pairs...", total=len(pretoken_freqs))
        for pretoken_bytes, freq in pretoken_freqs.items():
            if len(pretoken_bytes) < 2:
                byte_pairs.append(None)
                pbar.advance(task_id)
                continue

            start_idx = len(byte_pairs)
            for i in range(len(pretoken_bytes) - 1):
                pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
                byte_pair_counts[pair] += freq
                byte_pairs.append((pair, freq))
                bp_indices[pair].append(start_idx + i)

            byte_pairs.append(None)
            pbar.advance(task_id)

    del pretoken_freqs

    # Build heap
    logger.info("Building heap...")
    heap = [_PairEntry(count, pair) for pair, count in byte_pair_counts.items()]
    heapq.heapify(heap)

    logger.info(f"Total pairs in structure: {len(byte_pairs):,}")
    logger.info(f"Unique pair types: {len(byte_pair_counts):,}")

    # Step 3: BPE merging
    logger.info("Starting BPE merges...")
    merges: list[tuple[bytes, bytes]] = []
    deleted: set[int] = set()

    num_merges_needed = vocab_size - len(vocab)

    with ProgressBar() as pbar:
        task_id = pbar.add_task("[green]BPE Merging...", total=num_merges_needed)

        while len(vocab) < vocab_size:
            # Find the most common pair (with valid count)
            most_common_pair: Optional[tuple[bytes, bytes]] = None
            while heap:
                entry = heapq.heappop(heap)
                current_count = byte_pair_counts.get(entry.pair, 0)
                if current_count == entry.count and current_count > 0:
                    most_common_pair = entry.pair
                    break

            if most_common_pair is None:
                break

            # Create new merged token
            new_byte = most_common_pair[0] + most_common_pair[1]
            vocab[next_id] = new_byte
            next_id += 1
            merges.append(most_common_pair)

            # Track which pairs need heap updates
            updated_pairs: dict[tuple[bytes, bytes], int] = {}

            # Process all occurrences of this pair
            for occ_index in bp_indices[most_common_pair]:
                if occ_index in deleted:
                    continue

                entry = byte_pairs[occ_index]
                if entry is None or entry[0] != most_common_pair:
                    continue

                freq = entry[1]

                # Update previous pair (if exists)
                if occ_index > 0:
                    prev_index = occ_index - 1
                    while prev_index >= 0 and prev_index in deleted:
                        prev_index -= 1

                    if prev_index >= 0:
                        prev_entry = byte_pairs[prev_index]
                        if prev_entry is not None:
                            old_pair = prev_entry[0]
                            new_pair = (old_pair[0], new_byte)

                            byte_pair_counts[old_pair] -= freq
                            updated_pairs[old_pair] = byte_pair_counts[old_pair]

                            byte_pairs[prev_index] = (new_pair, freq)

                            byte_pair_counts[new_pair] += freq
                            updated_pairs[new_pair] = byte_pair_counts[new_pair]
                            bp_indices[new_pair].append(prev_index)

                # Update next pair (if exists)
                if occ_index < len(byte_pairs) - 1:
                    next_index = occ_index + 1
                    while next_index < len(byte_pairs) and next_index in deleted:
                        next_index += 1

                    if next_index < len(byte_pairs):
                        next_entry = byte_pairs[next_index]
                        if next_entry is not None:
                            old_pair = next_entry[0]
                            new_pair = (new_byte, old_pair[1])

                            byte_pair_counts[old_pair] -= freq
                            updated_pairs[old_pair] = byte_pair_counts[old_pair]

                            byte_pairs[next_index] = (new_pair, freq)

                            byte_pair_counts[new_pair] += freq
                            updated_pairs[new_pair] = byte_pair_counts[new_pair]
                            bp_indices[new_pair].append(next_index)

                # Mark current position as deleted
                deleted.add(occ_index)
                byte_pair_counts[most_common_pair] -= freq
                updated_pairs[most_common_pair] = byte_pair_counts[most_common_pair]

            # Push updated pairs to heap
            for pair, count in updated_pairs.items():
                if count > 0:
                    heapq.heappush(heap, _PairEntry(count, pair))

            pbar.advance(task_id)

    logger.info(f"BPE training complete! Final vocabulary size: {len(vocab)}")
    return vocab, merges


if __name__ == "__main__":
    path = "/home/yangty/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"Merges: {len(merges)}")

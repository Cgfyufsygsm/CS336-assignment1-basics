import heapq
import os

from tqdm import tqdm
from cs336_basics.pretokenization import pretokenize

def _iter_pairs(token_seq: tuple[bytes, ...]):
    for i in range(len(token_seq) - 1):
        yield (token_seq[i], token_seq[i + 1])

def _merge_pair_in_token_seq(
    token_seq: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]
) -> tuple[tuple[bytes, ...], bool]:
    merged_token = pair_to_merge[0] + pair_to_merge[1]
    new_token_seq = []
    i = 0
    changed = False
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == pair_to_merge:
            new_token_seq.append(merged_token)
            i += 2
            changed = True
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    if not changed:
        return token_seq, False
    return tuple(new_token_seq), True

def _update_counts_for_seq(
    pair_counts: dict[tuple[bytes, bytes], int],
    heap: list[tuple[int, tuple[bytes, bytes]]],
    token_seq: tuple[bytes, ...],
    count: int,
    delta: int,
) -> None:
    scale = delta * count
    for pair in _iter_pairs(token_seq):
        new_count = pair_counts.get(pair, 0) + scale
        if new_count:
            pair_counts[pair] = new_count
            heapq.heappush(heap, (-new_count, pair))
        else:
            pair_counts.pop(pair, None)

def _add_seq_to_index(
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    token_seq: tuple[bytes, ...],
) -> None:
    for pair in _iter_pairs(token_seq):
        pair_to_words.setdefault(pair, set()).add(token_seq)

def _remove_seq_from_index(
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    token_seq: tuple[bytes, ...],
) -> None:
    for pair in _iter_pairs(token_seq):
        words = pair_to_words.get(pair)
        if words is None:
            continue
        words.discard(token_seq)
        if not words:
            pair_to_words.pop(pair, None)

def _init_pair_state(word_tokens: dict[tuple[bytes, ...], int]):
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for token_seq, count in word_tokens.items():
        for pair in _iter_pairs(token_seq):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_words.setdefault(pair, set()).add(token_seq)
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return pair_counts, pair_to_words, heap

def _next_best_pair(
    pair_counts: dict[tuple[bytes, bytes], int],
    heap: list[tuple[int, tuple[bytes, bytes]]],
):
    while heap:
        neg_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count == 0 or -neg_count != count:
            continue
        # Resolve ties by lexicographically largest pair.
        candidates = [pair]
        while heap and heap[0][0] == neg_count:
            _, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count == 0 or -neg_count != count:
                continue
            candidates.append(pair)
        best_pair = max(candidates)
        for pair in candidates:
            if pair != best_pair:
                heapq.heappush(heap, (neg_count, pair))
        return best_pair, -neg_count
    return None, 0

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    print("Step 1/3: Pretokenizing input file...")
    pre_tokens = pretokenize(input_path, special_tokens)
    print(f"✓ Found {len(pre_tokens)} unique pretokens")

    print("\nStep 2/3: Initializing vocabulary...")
    merges = []

    word_tokens = {}
    for pretoken, count in pre_tokens.items():
        token_seq = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        word_tokens[token_seq] = word_tokens.get(token_seq, 0) + count

    vocab = {}
    next_id = 0

    # add special tokens to vocab
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    # add single byte tokens to vocab
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    print(f"✓ Initial vocabulary size: {len(vocab)} (special tokens + 256 bytes)")

    print(f"\nStep 3/3: Learning BPE merges (target vocab size: {vocab_size})...")
    pair_counts, pair_to_words, heap = _init_pair_state(word_tokens)

    num_merges_needed = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges_needed, desc="BPE merges", unit="merge")

    while len(vocab) < vocab_size:
        best_pair, _ = _next_best_pair(pair_counts, heap)
        if best_pair is None:
            break
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1
        merges.append(best_pair)

        impacted_words = list(pair_to_words.get(best_pair, ()))
        if not impacted_words:
            break
        new_word_counts: dict[tuple[bytes, ...], int] = {}
        for token_seq in impacted_words:
            count = word_tokens.pop(token_seq, None)
            if count is None:
                continue
            new_seq, changed = _merge_pair_in_token_seq(token_seq, best_pair)
            if not changed:
                word_tokens[token_seq] = count
                continue
            _update_counts_for_seq(pair_counts, heap, token_seq, count, -1)
            _remove_seq_from_index(pair_to_words, token_seq)
            new_word_counts[new_seq] = new_word_counts.get(new_seq, 0) + count
        for new_seq, count in new_word_counts.items():
            _update_counts_for_seq(pair_counts, heap, new_seq, count, 1)
            _add_seq_to_index(pair_to_words, new_seq)
            word_tokens[new_seq] = word_tokens.get(new_seq, 0) + count

        pbar.update(1)

    pbar.close()
    print(f"✓ BPE training complete! Final vocabulary size: {len(vocab)}")

    return vocab, merges


if __name__ == "__main__":
    path = "/home/yangty/CS336/assignment1-basics/test.txt"
    vocab_size = 263
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print("Vocab:", vocab)
    print("Merges:", merges)
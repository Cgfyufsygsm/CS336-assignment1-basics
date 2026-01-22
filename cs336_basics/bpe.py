import os

from cs336_basics.pretokenization import pretokenize

def get_pair_counts(word_tokens: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Given a dictionary mapping word token sequences to counts,
    return a dictionary mapping adjacent token pairs to their counts.
    """
    pair_counts = {}
    for token_seq, count in word_tokens.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts

def merge_pair_in_token_seq(token_seq: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Given a token sequence and a pair of tokens to merge,
    return a new token sequence with all occurrences of the pair merged.
    """
    merged_token = pair_to_merge[0] + pair_to_merge[1]
    new_token_seq = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == pair_to_merge:
            new_token_seq.append(merged_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    return tuple(new_token_seq)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    pre_tokens = pretokenize(input_path, special_tokens)
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
    
    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(word_tokens)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1
        merges.append(best_pair)
        word_tokens = {
            merge_pair_in_token_seq(token_seq, best_pair): count
            for token_seq, count in word_tokens.items()
        }

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
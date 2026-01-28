import json
import multiprocessing
import os
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        if special_tokens is not None:
            next_id = max(vocab.keys()) + 1
            for token in special_tokens:
                if token.encode("utf-8") not in self.byte_to_id:
                    self.vocab[next_id] = token.encode("utf-8")
                    self.byte_to_id[token.encode("utf-8")] = next_id
                    next_id += 1
        self.special_tokens = special_tokens if special_tokens is not None else []
    
    @classmethod
    def from_file(cls,
                  vocab_path: str | os.PathLike,
                  merges_path: str | os.PathLike,
                  special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode("utf-8") for k, v in vocab_json.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
        merges = [(token1.encode("latin-1"), token2.encode("latin-1")) for token1, token2 in merges_json]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def generate_bpe(self, text: str, res: list[int]) -> None:
        """
        Given an input text, apply pretokenization, apply BPE merges to generate token IDs,
        appending them to the `res` list.
        """
        if not text:
            return # return on empty string

        for pretoken_match in re.finditer(PAT, text):
            pretoken = pretoken_match.group(0)
            tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
            if len(tokens) >= 2:
                while True:
                    pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
                    best_pair = None
                    best_rank = None
                    for pair in pairs:
                        rank = self.merge_ranks.get(pair)
                        if rank is None:
                            continue
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    merged = []
                    i = 0
                    while i < len(tokens):
                        if (
                            i < len(tokens) - 1
                            and tokens[i] == best_pair[0]
                            and tokens[i + 1] == best_pair[1]
                        ):
                            merged.append(tokens[i] + tokens[i + 1])
                            i += 2
                        else:
                            merged.append(tokens[i])
                            i += 1
                    tokens = merged

            for token in tokens:
                if token in self.byte_to_id:
                    res.append(self.byte_to_id[token])
                else:
                    raise ValueError(f"Token {token} not in vocabulary.")

    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        res = []
        if not self.special_tokens:
            self.generate_bpe(text, res)
            return res
        pattern = "|".join(sorted(map(re.escape, self.special_tokens), key=len, reverse=True))
        pos = 0
        for m in re.finditer(pattern, text):
            self.generate_bpe(text[pos:m.start()], res)
            res.append(self.byte_to_id[m.group(0).encode("utf-8")])
            pos = m.end()

        self.generate_bpe(text[pos:], res)
        return res

    def _encode_batch_worker(self, chunk: list[str]) -> list[list[int]]:
        """Worker function to encode a batch of texts."""
        return [self.encode(text) for text in chunk]

    def _parallel_encode(self, chunk: list[str], pool_size: int) -> Iterator[int]:
        """Parallel encode a batch of texts."""
        if pool_size == 1:
            # Single process mode
            for text in chunk:
                yield from self.encode(text)
        else:
            # Multi-process mode
            batch_size = (len(chunk) + pool_size - 1) // pool_size
            mini_batches = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]

            with multiprocessing.Pool(processes=pool_size) as pool:
                results_of_lists = pool.map(self._encode_batch_worker, mini_batches)

            # Stream results
            for worker_result in results_of_lists:
                for ids in worker_result:
                    yield from ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
        chunk_size: int = 10240,
        pool_size: int | None = None,
    ) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable: An iterable of strings to encode (e.g., file handle)
            chunk_size: Number of strings to accumulate before parallel processing (default: 10240)
            pool_size: Number of processes to use (default: cpu_count - 1, or 1 if cpu_count unavailable)

        Yields:
            Token IDs one at a time
        """
        if pool_size is None:
            pool_size = max(1, (os.cpu_count() or 2) - 1)

        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield from self._parallel_encode(chunk, pool_size)
                chunk = []

        # Process remaining items
        if chunk:
            yield from self._parallel_encode(chunk, pool_size)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

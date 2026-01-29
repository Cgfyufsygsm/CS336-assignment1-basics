import json
import multiprocessing
from multiprocessing.pool import Pool
import os
import regex as re
from typing import Iterable, Iterator

from cs336_basics.tokenizer.utils import decode_token_gpt2

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = re.compile(PAT)

_WORKER_TOKENIZER = None


def _init_worker(tokenizer: "Tokenizer") -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_batch_worker_global(chunk: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker is not initialized.")
    return [_WORKER_TOKENIZER.encode(text) for text in chunk]

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
        if self.special_tokens:
            pattern = "|".join(sorted(map(re.escape, self.special_tokens), key=len, reverse=True))
            self._special_re = re.compile(pattern)
        else:
            self._special_re = None
    
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
        if not vocab_json or not all(isinstance(v, int) for v in vocab_json.values()):
            raise ValueError("Expected GPT-2 style vocab JSON (token -> id).")
        vocab = {v: decode_token_gpt2(k) for k, v in vocab_json.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
        merges = [(decode_token_gpt2(token1), decode_token_gpt2(token2)) for token1, token2 in merges_json]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def generate_bpe(self, text: str, res: list[int]) -> None:
        """
        Given an input text, apply pretokenization, apply BPE merges to generate token IDs,
        appending them to the `res` list.
        """
        if not text:
            return # return on empty string

        merge_ranks = self.merge_ranks
        byte_to_id = self.byte_to_id
        for pretoken_match in PRETOKEN_RE.finditer(text):
            pretoken = pretoken_match.group(0)
            tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
            if len(tokens) >= 2:
                while True:
                    best_pair = None
                    best_rank = None
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        rank = merge_ranks.get(pair)
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
                token_id = byte_to_id.get(token)
                if token_id is None:
                    raise ValueError(f"Token {token} not in vocabulary.")
                res.append(token_id)

    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        res = []
        if not self._special_re:
            self.generate_bpe(text, res)
            return res
        pos = 0
        for m in self._special_re.finditer(text):
            self.generate_bpe(text[pos:m.start()], res)
            res.append(self.byte_to_id[m.group(0).encode("utf-8")])
            pos = m.end()

        self.generate_bpe(text[pos:], res)
        return res

    def _encode_batch_worker(self, chunk: list[str]) -> list[list[int]]:
        """Worker function to encode a batch of texts."""
        return [self.encode(text) for text in chunk]

    def _parallel_encode(
        self,
        chunk: list[str],
        pool_size: int | None = None,
        pool: Pool | None = None,
    ) -> Iterator[int]:
        """Parallel encode a batch of texts."""
        if pool_size is None:
            pool_size = max(1, (os.cpu_count() or 2) - 1)
        if pool is None and pool_size == 1:
            # Single process mode
            for text in chunk:
                yield from self.encode(text)
            return

        # Multi-process mode
        batch_size = (len(chunk) + pool_size - 1) // pool_size
        mini_batches = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
        if pool is None:
            with multiprocessing.Pool(
                processes=pool_size,
                initializer=_init_worker,
                initargs=(self,),
            ) as local_pool:
                for worker_result in local_pool.imap(_encode_batch_worker_global, mini_batches):
                    for ids in worker_result:
                        yield from ids
            return

        use_worker_state = getattr(pool, "_initializer", None) is _init_worker
        worker_fn = _encode_batch_worker_global if use_worker_state else self._encode_batch_worker
        for worker_result in pool.imap(worker_fn, mini_batches):
            for ids in worker_result:
                yield from ids

    def create_pool(self, pool_size: int) -> Pool:
        return multiprocessing.Pool(processes=pool_size, initializer=_init_worker, initargs=(self,))

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
        chunk = []
        if pool_size == 1:
            for item in iterable:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    yield from self._parallel_encode(chunk, pool_size=1)
                    chunk = []
            if chunk:
                yield from self._parallel_encode(chunk, pool_size=1)
            return

        with self.create_pool(pool_size) as pool:
            for item in iterable:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    yield from self._parallel_encode(chunk, pool_size=pool_size, pool=pool)
                    chunk = []
            if chunk:
                yield from self._parallel_encode(chunk, pool_size=pool_size, pool=pool)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

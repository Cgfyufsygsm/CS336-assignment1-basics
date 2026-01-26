import json
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

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                token1, token2 = line.rstrip().split(" ")
                merges.append((token1.encode("utf-8"), token2.encode("utf-8")))
        
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
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

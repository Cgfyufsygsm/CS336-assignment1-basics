"""Pretokenization for BPE training with multiprocessing support."""
import functools
import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

from cs336_basics.utils import get_logger, timer, ProgressBar

# Pre-compile regex pattern for performance
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PREPATTERN = re.compile(PAT)
BYTES_LOOKUP = [bytes([i]) for i in range(256)]

logger = get_logger(__name__)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_text(text: str) -> Counter[tuple[bytes, ...]]:
    """Pretokenize a single text using generator chain for efficiency."""
    # Generator chain: regex match -> encode -> byte tuple
    raw_bytes_gen = (m.group().encode() for m in re.finditer(PREPATTERN, text))
    pretoken_gen = (tuple(BYTES_LOOKUP[b] for b in raw) for raw in raw_bytes_gen)
    return Counter(pretoken_gen)


def _process_chunk(
    pos: tuple[int, int],
    path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """Process a single chunk of the file."""
    start, end = pos
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        pattern = '|'.join(map(re.escape, special_tokens))
        docs = re.split(pattern, data)
    else:
        docs = [data]

    counter: Counter[tuple[bytes, ...]] = Counter()
    for doc in docs:
        counter.update(_pretokenize_text(doc))
    return counter


@timer
def pretokenize(
    path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None
) -> Counter[tuple[bytes, ...]]:
    """
    Pre-tokenize the file at `path` using multiple processes.
    Returns a Counter of pre-token frequencies as tuple[bytes, ...].
    """
    if num_processes is None:
        num_processes = max(1, (os.cpu_count() or 4) - 1)

    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = list(zip(boundaries[:-1], boundaries[1:]))

    logger.info(f"Processing {len(tasks)} chunks with {num_processes} workers")
    total_counts: Counter[tuple[bytes, ...]] = Counter()

    with ProgressBar() as pbar:
        task_id = pbar.add_task("[cyan]Pretokenizing...", total=len(tasks))
        with Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(
                functools.partial(_process_chunk, path=path, special_tokens=special_tokens),
                tasks
            )
            for counter in results:
                total_counts.update(counter)
                pbar.advance(task_id)

    logger.info(f"Found {len(total_counts):,} unique pretokens")
    return total_counts


if __name__ == "__main__":
    path = "/home/yangty/CS336/assignment1-basics/data/owt_valid.txt"
    result = pretokenize(path, ["<|endoftext|>"])
    logger.info(f"Unique pretokens: {len(result)}")
    logger.info("Top 10:")
    for tok, cnt in result.most_common(10):
        logger.info(f"  {tok} -> {cnt}")

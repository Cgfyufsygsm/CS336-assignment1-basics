import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")
        
        pattern = '|'.join(map(re.escape, special_tokens))

        docs = re.split(pattern, data)

        counter = Counter()
        for doc in docs:
            counter.update(m.group(0) for m in re.finditer(PAT, doc))
        return counter

def pretokenize(path: str | os.PathLike,
                special_tokens: list[str],
                num_processes=None
) -> Counter:
    """
    Pre-tokenize the file at `path` using multiple processes.
    Returns a Counter of pre-token frequencies.
    """
    if num_processes is None:
        num_processes = os.cpu_count() or 4

    # Create more chunks than processes for better load balancing
    desired_chunks = num_processes * 8

    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, b"<|endoftext|>")

    tasks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((path, start, end, special_tokens))

    print(f"  Processing {len(tasks)} chunks with {num_processes} workers...")
    total_counts = Counter()
    with Pool(processes=num_processes) as pool:
        for counter in tqdm(
            pool.imap_unordered(process_chunk, tasks),
            total=len(tasks),
            desc="  Chunks",
            unit="chunk",
        ):
            total_counts.update(counter)

    return total_counts

## Usage
if __name__ == "__main__":
    path = "/home/yangty/CS336/assignment1-basics/data/owt_valid.txt"
    with open(path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    tasks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((path, start, end, ["<|endoftext|>"]))

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, tasks)
        # Combine results from all chunks
    total_counts = Counter()
    for counter in results:
        total_counts.update(counter)
    print(total_counts.most_common(10))

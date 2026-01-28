import argparse
import io
import os
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.utils import ProgressBar


def _parse_special_tokens(value: str) -> list[str]:
    if not value:
        return []
    return [tok for tok in value.split(",") if tok]


def encode_to_memmap(
    input_path: Path,
    vocab_path: Path,
    merges_path: Path,
    output_path: Path,
    dtype: str,
    special_tokens: list[str],
) -> None:
    if output_path.suffix != ".npy":
        raise ValueError("Output must be a .npy file.")
    tokenizer = Tokenizer.from_file(vocab_path, merges_path, special_tokens=special_tokens)
    max_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    dtype_info = np.iinfo(np.dtype(dtype))
    if max_id > dtype_info.max:
        raise ValueError(
            f"Tokenizer vocab has id {max_id}, which does not fit in dtype {dtype} "
            f"(max {dtype_info.max})."
        )

    # Single-pass encoding with byte-based progress for ETA.
    print(f"Encoding {input_path}...")
    file_size = input_path.stat().st_size
    chunk_size = 10240
    pool_size = max(1, (os.cpu_count() or 2) - 1)

    def token_generator():
        with open(input_path, "rb") as raw, ProgressBar() as progress:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8")
            task_id = progress.add_task("Encoding tokens", total=file_size)
            last_pos = raw.tell()
            chunk: list[str] = []

            for line in text_stream:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield from tokenizer._parallel_encode(chunk, pool_size)
                    chunk.clear()
                    new_pos = raw.tell()
                    progress.update(task_id, advance=new_pos - last_pos)
                    last_pos = new_pos

            if chunk:
                yield from tokenizer._parallel_encode(chunk, pool_size)
                new_pos = raw.tell()
                progress.update(task_id, advance=new_pos - last_pos)

    arr = np.fromiter(token_generator(), dtype=np.dtype(dtype))

    # Save
    print(f"Writing {arr.size:,} tokens to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)

    print(f"Wrote {arr.size:,} tokens to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text dataset into token IDs (memmap).")
    parser.add_argument("--input", required=True, type=Path, help="Path to input text file.")
    parser.add_argument(
        "--vocab",
        required=True,
        type=Path,
        help="Path to tokenizer vocab.json (GPT-2 byte-encoded, token -> id).",
    )
    parser.add_argument(
        "--merges",
        required=True,
        type=Path,
        help="Path to tokenizer merges.json (GPT-2 byte-encoded token pairs).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to output .npy file.")
    parser.add_argument("--dtype", default="uint16", help="Numpy dtype for token IDs.")
    parser.add_argument(
        "--special_tokens",
        default="<|endoftext|>",
        help="Comma-separated list of special tokens.",
    )
    args = parser.parse_args()

    encode_to_memmap(
        input_path=args.input,
        vocab_path=args.vocab,
        merges_path=args.merges,
        output_path=args.output,
        dtype=args.dtype,
        special_tokens=_parse_special_tokens(args.special_tokens),
    )


if __name__ == "__main__":
    main()

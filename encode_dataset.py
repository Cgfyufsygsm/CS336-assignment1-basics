import argparse
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

    # Single-pass encoding: accumulate tokens in a list first
    print(f"Encoding {input_path}...")
    tokens = []

    with open(input_path, "r", encoding="utf-8") as f, ProgressBar() as progress:
        # Use indeterminate progress bar since we don't know total tokens yet
        task_id = progress.add_task("Encoding tokens", total=None)
        batch_size = 100000  # Update progress every 100k tokens
        batch_count = 0

        for token_id in tokenizer.encode_iterable(f):
            tokens.append(token_id)
            batch_count += 1

            if batch_count >= batch_size:
                progress.update(task_id, advance=batch_count)
                batch_count = 0

        # Update final batch
        if batch_count > 0:
            progress.update(task_id, advance=batch_count)

    # Convert to numpy array and save
    print(f"Converting {len(tokens):,} tokens to numpy array...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tokens, dtype=np.dtype(dtype))
    np.save(output_path, arr)

    print(f"Wrote {len(tokens):,} tokens to {output_path}")


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

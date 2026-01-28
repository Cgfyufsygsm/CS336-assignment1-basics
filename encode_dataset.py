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

    with open(input_path, "r", encoding="utf-8") as f:
        n_tokens = sum(1 for _ in tokenizer.encode_iterable(f))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.dtype(dtype), shape=(n_tokens,))

    with open(input_path, "r", encoding="utf-8") as f, ProgressBar() as progress:
        task_id = progress.add_task("Encoding tokens", total=n_tokens)
        i = 0
        for token_id in tokenizer.encode_iterable(f):
            arr[i] = token_id
            i += 1
            progress.update(task_id, advance=1)

    arr.flush()
    print(f"Wrote {n_tokens} tokens to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text dataset into token IDs (memmap).")
    parser.add_argument("--input", required=True, type=Path, help="Path to input text file.")
    parser.add_argument("--vocab", required=True, type=Path, help="Path to tokenizer vocab.json.")
    parser.add_argument("--merges", required=True, type=Path, help="Path to tokenizer merges.txt.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output .bin file.")
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

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from config import AppConfig
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.tokenizer.tokenizer import Tokenizer


def _load_base_vocab_size(vocab_path: str | Path) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    if not vocab_json or not all(isinstance(v, int) for v in vocab_json.values()):
        raise ValueError("Expected GPT-2 style vocab JSON (token -> id).")
    return len(vocab_json)


def _load_model_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError("Checkpoint must be a state dict or include a model_state_dict entry.")


def _parse_args() -> argparse.Namespace:
    cfg = AppConfig()
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM.")

    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt).")
    parser.add_argument("--vocab_json", required=True, help="Path to vocab JSON.")
    parser.add_argument("--merges", required=True, help="Path to merges JSON.")

    parser.add_argument("--prompt", default="", help="Prompt text to condition on.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eos_token", default="<|endoftext|>")
    parser.add_argument("--special_token", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--device", default=cfg.data.device)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=cfg.model.context_length)
    parser.add_argument("--d_model", type=int, default=cfg.model.d_model)
    parser.add_argument("--n_layers", type=int, default=cfg.model.n_layers)
    parser.add_argument("--n_heads", type=int, default=cfg.model.n_heads)
    parser.add_argument("--d_ff", type=int, default=cfg.model.d_ff)
    parser.add_argument("--rope_theta", type=float, default=100000.0)
    tie_group = parser.add_mutually_exclusive_group()
    tie_group.add_argument("--tie_embeddings", action="store_true", default=cfg.model.tie_embeddings)
    tie_group.add_argument("--no_tie_embeddings", action="store_false", dest="tie_embeddings")

    return parser.parse_args()


def _build_tokenizer(
    vocab_json: str | Path,
    merges: str | Path,
    special_tokens: Sequence[str] | None,
) -> Tokenizer:
    return Tokenizer.from_file(vocab_json, merges, special_tokens=list(special_tokens) if special_tokens else None)


def main() -> None:
    args = _parse_args()
    base_vocab_size = _load_base_vocab_size(args.vocab_json)

    special_tokens = list(args.special_token) if args.special_token else []
    if args.eos_token and args.eos_token not in special_tokens:
        special_tokens.append(args.eos_token)
    tokenizer = _build_tokenizer(args.vocab_json, args.merges, special_tokens or None)

    if args.vocab_size is None:
        args.vocab_size = len(tokenizer.vocab)
        if len(tokenizer.vocab) != base_vocab_size:
            print(
                "Warning: special tokens expanded the vocab; "
                "ensure the checkpoint was trained with this vocab size."
            )

    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layers,
        rope_theta=args.rope_theta,
    )
    model.load_state_dict(_load_model_state(args.checkpoint))
    if args.tie_embeddings:
        model.linear.weight = model.token_embedding.weight
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    eos_id = None
    if args.eos_token is not None:
        eos_bytes = args.eos_token.encode("utf-8")
        if eos_bytes not in tokenizer.byte_to_id:
            raise ValueError("eos_token not found in tokenizer vocabulary.")
        eos_id = tokenizer.byte_to_id[eos_bytes]

    output_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
        rng=None,
    )
    output_text = tokenizer.decode(output_ids.detach().cpu().tolist())
    print(output_text)


if __name__ == "__main__":
    main()

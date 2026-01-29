import torch
from torch import Tensor

from cs336_basics.nn.util import softmax


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """
    Scale logits by temperature. Lower temperature sharpens the distribution.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for scaling")
    return logits / temperature


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """
    Apply nucleus (top-p) filtering to logits.
    Keeps the smallest set of tokens whose cumulative probability >= top_p.
    """
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        top_idx = logits.argmax(dim=-1, keepdim=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(-1, top_idx, False)
        return logits.masked_fill(mask, -float("inf"))

    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    mask = torch.zeros_like(cutoff).scatter(-1, sorted_idx, cutoff)
    return logits.masked_fill(mask, -float("inf"))


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: torch.Generator | None = None,
) -> Tensor:
    """
    Sample token ids from logits (shape: [batch, vocab] or [vocab]).
    """
    squeeze_out = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_out = True

    if temperature <= 0:
        next_ids = logits.argmax(dim=-1)
        return next_ids.squeeze(0) if squeeze_out else next_ids

    scaled = apply_temperature(logits, temperature)
    filtered = top_p_filter(scaled, top_p)
    probs = softmax(filtered, dim=-1)
    next_ids = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)
    return next_ids.squeeze(0) if squeeze_out else next_ids


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: Tensor | list[int],
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    context_length: int | None = None,
    rng: torch.Generator | None = None,
) -> Tensor:
    """
    Autoregressively generate tokens from a prompt.
    """
    device = next(model.parameters()).device
    if torch.is_tensor(prompt_ids):
        prompt = prompt_ids.to(device=device, dtype=torch.long)
    else:
        prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long)

    squeeze_out = False
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
        squeeze_out = True

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    generated = prompt
    finished = None
    eos_tensor = None
    if eos_id is not None:
        finished = torch.zeros(generated.size(0), device=device, dtype=torch.bool)
        eos_tensor = torch.tensor(eos_id, device=device, dtype=torch.long)

    for _ in range(max_new_tokens):
        if context_length is not None and generated.size(1) > context_length:
            input_ids = generated[:, -context_length:]
        else:
            input_ids = generated

        logits = model(input_ids)
        next_logits = logits[:, -1, :]
        next_ids = sample_next_token(next_logits, temperature=temperature, top_p=top_p, rng=rng)

        if finished is not None:
            next_ids = torch.where(finished, eos_tensor, next_ids)
            finished |= next_ids == eos_id

        generated = torch.cat([generated, next_ids.unsqueeze(-1)], dim=-1)
        if finished is not None and torch.all(finished):
            break

    return generated.squeeze(0) if squeeze_out else generated

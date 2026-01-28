from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a 1D token ID array.

    Returns:
        Tuple of LongTensors (x, y) with shape (batch_size, context_length).
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D numpy array of token IDs")
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("dataset length must be greater than context_length")

    starts = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([dataset[i : i + context_length] for i in starts], axis=0)
    y_np = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts], axis=0)

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y

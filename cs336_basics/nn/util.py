import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Given an input tensor `x`, return the softmax applied along dimension `dim`.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
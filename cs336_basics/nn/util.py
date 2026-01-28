import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Given an input tensor `x`, return the softmax applied along dimension `dim`.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def CrossEntropyLoss(input: torch.Tensor, target: torch.Tensor):
    z = input - input.max(dim=-1, keepdim=True).values
    log_denom = torch.logsumexp(z, dim=-1, keepdim=True)
    z_true = z.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    loss = (-z_true + log_denom).mean()
    return loss
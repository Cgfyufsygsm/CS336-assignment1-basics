import torch
import torch.nn as nn
from cs336_basics.nn.layers import Linear

def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    Given an input tensor `x`, return the SiLU activation applied elementwise.
    SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = SiLU(self.linear1(x))
        value = self.linear3(x)
        return self.linear2(gate * value)
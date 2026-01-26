import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, x):
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.gain).to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert(d_k % 2 == 0), "d_k must be even for Rotary Positional Embedding."
        half = d_k // 2
        k = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = torch.pow(theta, -2 * k / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angle = positions[:, None] * inv_freq[None, :]
        cos_cached, sin_cached = torch.cos(angle), torch.sin(angle)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pair = x.view(*x.shape[:-1], -1, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        cos_pos = getattr(self, "cos_cached")[token_positions]
        sin_pos = getattr(self, "sin_cached")[token_positions]
        x_even_rot = x_even * cos_pos - x_odd * sin_pos
        x_odd_rot = x_even * sin_pos + x_odd * cos_pos
        x_pair = torch.stack((x_even_rot, x_odd_rot), dim=-1)
        return x_pair.view(*x.shape[:-1], self.d_k)
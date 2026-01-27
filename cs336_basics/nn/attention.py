import math
import torch
import torch.nn as nn
from cs336_basics.nn.util import softmax
from cs336_basics.nn.layers import Linear
from cs336_basics.nn.layers import RotaryPositionalEmbedding

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """
    scaled_dot_product_attention
    
    :param q: (batch_size, ..., seq_len, d_k)
    :param k: (batch_size, ..., seq_len, d_k)
    :param v: (batch_size, ..., seq_len, d_v)
    :param mask: (batch_size, ..., seq_len, seq_len) or None
    """
    d_k = q.shape[-1]
    score = q @ k.transpose(-2, -1) # (batch_size, ..., seq_len, seq_len)
    scores = score / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    attn = softmax(scores, dim=-1)
    return attn @ v

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # Ensure divisibility
        self.head_dim = d_model // num_heads

        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rope = rope


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x) # (..., seq_len, d_model)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        # -2 is num_heads, -3 is seq_len, so need to transpose
        k = k.view(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        v = v.view(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool, device=x.device)) # (seq_len, seq_len)
        # score[i, j] means how i attends to j, so we want to mask out j > i
        # so for j > i, we set mask[i, j] = False, thus lower triangular

        if self.rope is not None:
            assert token_positions is not None, "token_positions must be provided when using RoPE."
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attn_output = scaled_dot_product_attention(q, k, v, mask=mask) # num_heads will be treated as batch dimension
        attn_output = attn_output.transpose(-3, -2).contiguous() # (..., seq_len, num_heads, head_dim)
        attn_output = attn_output.view(*attn_output.shape[:-2], self.d_model) # (..., seq_len, d_model)
        output = self.W_O(attn_output) # (..., seq_len, d_model)
        return output
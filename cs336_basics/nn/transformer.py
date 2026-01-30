import torch
import torch.nn as nn

from cs336_basics.nn.attention import MultiheadSelfAttention
from cs336_basics.nn.layers import Embedding, Linear
from cs336_basics.nn.layers import RotaryPositionalEmbedding
from cs336_basics.nn.network import SwiGLU
from cs336_basics.nn.decoding import generate_tokens
from cs336_basics.nn.util import softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 100000.0, max_seq_len: int = 2048):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len=max_seq_len)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        y = x + self.attn(x)
        return y + self.ffn(y)
    
class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float = 100000.0):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.attention_blocks = nn.Sequential(*[TransformerBlock(
            d_model,
            num_heads,
            d_ff,
            theta=rope_theta,
            max_seq_len=context_length
        ) for _ in range(num_layers)])
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        x = self.attention_blocks(x)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor | list[int],
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        context_length: int | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        return generate_tokens(
            self,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
            temperature=temperature,
            top_p=top_p,
            context_length=context_length,
            rng=rng,
        )

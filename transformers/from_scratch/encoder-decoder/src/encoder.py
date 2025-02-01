"""A torch transformer for learning encoder from scratch"""

import math
import torch
from torch import nn

from torch.nn.init import xavier_uniform_
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import SinusoidalEncoding


class TransformerEncoder(nn.Module):
    """Defines transformer encoders."""

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embed = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = SinusoidalEncoding(hidden_dim, max_len=5000)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor, src_padding_mask: torch.BoolTensor = None
    ):
        """
        Performs one encoder forward pass given input
        token ids and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param input_ids: Tensor containing input token
        ids. Shape: (N, S)
        :param src_padding_mask: An attention mask to
        ignore pad-tokens in the source input. Shape (N, S)
        :return: The encoder's final (contextualized)
        token embeddings. Shape: (N, S, E)
        """
        x = self.embed(input_ids) * math.sqrt(self.hidden_dim)  # (N, S, E)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, src_padding_mask=src_padding_mask)
        return x


class EncoderBlock(nn.Module):
    """Encoder block."""

    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        """Encode block module.

        Args:
            hidden_dim (int): _description_
            ff_dim (int): _description_
            num_heads (int): _description_
            dropout_p (float): _description_
        """
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.FloatTensor, src_padding_mask: torch.BoolTensor = None):
        """
        Performs one encoder *block* forward pass given the previous block's
        output and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param x: Tensor containing the output of the previous encoder block.
        Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens
        in the source input. Shape (N, S)
        :return: Updated intermediate encoder (contextualized) token
        embeddings. Shape: (N, S, E)
        """
        output = self.dropout1(
            self.self_mha.forward(x, src_padding_mask=src_padding_mask)
        )
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)
        return x

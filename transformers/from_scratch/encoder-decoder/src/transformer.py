"""A base transformer model"""

from typing import Optional
from torch import nn
from torch.nn.init import xavier_uniform_
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """The transformer model."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        max_decoding_length: int,
        vocab_size: int,
        padding_idx: int,
        bos_idx: int,
        dropout_p: float,
        tie_output_to_embedding: Optional[bool] = None,
    ):
        super().__init__()
        # Because the encoder embedding, and decoder embedding and
        # decoder pre-softmax transformation share embeddings
        # weights, initialize one here and pass it on.
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

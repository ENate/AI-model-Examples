"""Implementation of the decoder step."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import SinusoidalEncoding


class TransformerDecoder(nn.Module):
    """Defines the transformer decoder model."""

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        dropout_p: float,
        tie_output_to_embedding: Optional[bool] = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed = embedding
        self.positional_encoding = SinusoidalEncoding(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Note: a linear layer multiplies the input with a transpose of the
        # weight matrix, so no need to do that here.
        if tie_output_to_embedding:
            self.output_layer.weight = nn.Parameter(self.embed.weight)

    def _reset_parameters(self):
        """Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        input_tokens: torch.IntTensor,
        encoder_hidden_states: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs one decoder forward pass given encoder hidden states, the
        decoder input tokens and attention masks.
        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param input_tokens: Decoder input tokens. Shape: (N, T)
        :param encoder_hidden_states: The encoder's final (contextualized)
        token embeddings. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the
        source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the
        target input. Shape (T, T)
        :return: Un-normalized logits over the vocabulary for every token in
        the batch. Shape (N, T, V)
        """
        # (batch_size, sequence_length, hidden_dim)
        x = self.embed(input_tokens) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, src_padding_mask, future_mask)

        # (batch_size, sequence_length, vocab_size)
        logits = self.output_layer(x)
        return logits


class TransformerDecoderBlock(nn.Module):
    """Generate transformer block

    Args:
        nn (model): defines neural network model.
    """

    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs one decoder *block* forward pass given final encoder hidden
        states, the previous block's output, and
        attention masks.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param x: Previous decoder block's output. Shape: (N, T, E)
        :param encoder_hidden_states: The encoder's final (contextualized)
        token embeddings. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the
        source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the
        target input. Shape (T, T)
        :return: Updated, contextualized token embeddings. Shape (N, T, E)
        """

        # Self attention (with future masking during training)
        output = self.dropout1(self.self_mha.forward(x, future_mask=future_mask))
        x = self.layer_norm1(x + output)

        # Cross or encoder-decoder attention
        output = self.dropout2(
            self.cross_mha.forward(
                x,
                encoder_hidden_states=encoder_hidden_states,
                src_padding_mask=src_padding_mask,
            )
        )
        x = self.layer_norm2(x + output)

        # Feed forward layers
        output = self.dropout3(self.feed_forward(x))
        x = self.layer_norm3(x + output)
        return x

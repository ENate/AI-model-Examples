"""Implementation of the Multi-headed attention."""

from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self._reset_parameter()

    def _reset_parameter(self):
        """ Weight initialization
        taken from the UvA DL1 PyTorch Transformer tutorial.
        """
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None
    ):
        """Perform multi-head attention using one projection matrix.
        Self attention is performed when encoder_hidden_states
        is None, in which case input x represents encoder token embeddings.
        Otherwise, cross-attention is performed.
        In that case, input x represents the decoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality

        Args:
            :param x: Either encoder or decoder hidden states. Shape: (N, S or T, E)
            :param encoder_hidden_states: Encoder hidden states to perform cross-attention with. Shape: (N, S, E)
            :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
            Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
            Shape: (N, S)
            :param future_mask: Used for decoder self-attention to avoid any token i attending to a token >i, i.e. "peaking"
            Shape: (T, T).
            :return: Contextualized token embeddings. Shape depends on attention type. (N, S, E) for encoder self-attention
            and decoder cross-attention. (N, T, E) for decoder self-attention.
        """
        batch_size, sequence_length, hidden_dim = x.size()
        if encoder_hidden_states is None:
            q, k, v = self._self_attention_projection(x)
        else:
            q, k, v = self._cross_attention_projection(encoder_hidden_states, x)

        # Swap dimensions to (batch_size, n_heads, seq_len, qkv_dim). Required for the matrix multiplication below
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute (contextualized) value vector for each "head"
        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask, future_mask)

        # Concatenate contextualized value vectors from all heads
        values = values.reshape(batch_size, sequence_length, hidden_dim)

        # Linearly transform the concatenation of all heads' value vectors (8*64=512) to the original hidden dim (512)
        output = self.o_proj(values)
        return output

    def _self_attention_projection(self, x: torch.Tensor):
        """Project x and interpret the result as chunks that represent q, k and v vectors for every head.
        Input x can be encoder or decoder hidden states, depending on which one calls this MHA module.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param x (torch.Tensor): Encoder or decoder hidden states. (N, S or T, E)
        :return: query, key and value vectors. (N, S or T, H, E/H)
        """
        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(
        self, encoder_hidden_states: torch.Tensor,
        decoder_hidden_states: torch.Tensor
        ):
        """
        Projects hidden states into query vectors and encoder hidden states into key and value vectors.
        The columns of w_proj determine how much independent linear combination of the inputs we obtain -
        which we then interpret as heads and qkv vectors. Then we can simply split the weight matrix and
        project the decoder hidden states into q separately from projecting the encoder_hidden_states into k and v.
        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads


        Args:
            :param encoder_hidden_states (torch.Tensor): Shape: (N, S, E)
            :param decoder_hidden_states (torch.Tensor): Shape: (N, T, E)
            :return query vector Shape: (N, T, H, E/H) and key and value vectors both (N, S, H, E/H)

        """
        batch_size, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        batch_size, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

        #split weight matrix
        w_q, w_kv = self.qkv_proj.weight.split([hidden_dim, 2 * hidden_dim])

        # Project encoder_hidden_states into k's, and v's
        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv)
            .reshape(batch_size, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )
        # Project decoder hidden states into q's

        q = (
            F.linear(input=decoder_hidden_states, weight=w_q)
            .reshape(batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim)
        )
        return q, k, v

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None
    ):
        """
        For cross-attention, the sequence length of q and (k,v) may differ as q is
        projected from decoder hidden states and kv from encoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param q: Tensor stacking query vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param k: Tensor stacking key vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param v: Tensor stacking value vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)
        :param future_mask: Used for decoder self-attention to avoid any token i attending to a token >i, i.e. "peaking"
        Shape: (T, T).
        :return: values (N, H, S or T, E/H), attention scores (N, H, S or T, S or T)
        """
        # Compute attention logits. Dot product between each query and key vector, through one matrix multiplication.
        # Results in un-normalized attention scores for each position's query vector to each position's key vector
        # Result is (batch_size, num_heads, seq_length, seq_length)
        attn_logits = torch.matmul(q, torch.transpose(k, -2, -1),)
        # Scale logits by constant to create less spiky softmax distribution
        attn_logits = attn_logits / math.sqrt(q.size()[-1])
        # Apply attention mask (for pad tokens and future-masking in cross-attention)
        if src_padding_mask is not None or future_mask is not None:
            attn_logits = self.mask_logits(attn_logits, src_padding_mask, future_mask)  # type: ignore

        # Transform logits to attention probability distribution (one distribution per non-masked token index)
        attention = F.softmax(attn_logits, dim=-1)
        # Weighted sum of value vectors for each input token using attention scores -> new contextualized representation
        # (batch_size, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)
        return values, attention

    @staticmethod
    def mask_logits(
        logits: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None
    ):
        """Reshape masks to fit the shape of the logits and set all indices with "False" to -inf

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        Args:
            :param logits (torch.Tensor): Tensor containing attention logits. Shape: (N, H, S or T, S or T)
            :param src_padding_mask (Optional[torch.BoolTensor], optional): Used for encoder self-attention
            and cross-attention to handle pad tokens. Masks all incoming "connections" or "logits" from any
            token position to any pad token in a sequence. Shape: (N, S)Defaults to None.
            :param future_mask (Optional[torch.BoolTensor], optional): Used for decoder self-attention
            to avoid any token i attending to a token >i, i.e. "peaking" Shape: (T, T).. Defaults to None.
            :return: masked_logits (N, H, S or T, S or T)
        """
        if src_padding_mask is not None:
            masked_logits = logits.masked_fill(
                src_padding_mask[:, None, None, :] == 0, float("-inf")
            )
        if future_mask is not None:
            masked_logits = logits.masked_fill(future_mask == 0, float("-inf"))
        return masked_logits

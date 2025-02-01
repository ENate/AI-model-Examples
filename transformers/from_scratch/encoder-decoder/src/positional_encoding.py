import math
import torch


class SinusoidalEncoding(torch.nn.Module):
    """ Copied from this link:
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """


    def __init__(self, hidden_dim, max_len=8):
        """ Defines the hidden input dimension and max sentence
        length to expect.

        Args:
            hidden_dim (_type_): hidden input dimension
            max_len (_type_, optional): expected sentence max length. Defaults to 5000.
        """
        super().__init__()
        # Create matrix of dimension [seqLen, hidden_dim]
        pos_embed =  torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, hidden_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
    
    def forward(self, x):
        """
        Adds positional embeddings to token embeddings.
        N = batch size
        L = sequence length
        E = embedding dim

        :param x: token embeddings. Shape: (N, L, E)
        :return: token_embeddings + positional embeddings. Shape: (N, L, E)
        """
        x = x + self.pos_embed[:, : x.size(1)]
        return x
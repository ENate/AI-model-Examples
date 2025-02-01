"""The transformer test implementation."""

import random
import torch
import unittest
import numpy as np

from .src.utils import construct_future_mask
from .src.transformer import Transformer
from .src.vocabulary import Vocabulary


class TestTransformer(unittest.TestCase):
    """Test case for the transformer implementation.
    Args:
        unittest (_type_): represents the transformer test class
    """

    def test_transformer_inference(self):
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create (shared) Vocabulary and special token
        # given a dummy corpus
        corpus = [
            "Hello my name is Janice and I was born with the name Janice"
            "Dit is een Nederlandse zin"
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token_2_index.items())
        with torch.no_grad():
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                vocab_size=en_vocab_size,
                padding_idx=en_vocab.token_2_index[en_vocab.PAD],
                bos_idx=en_vocab.token_2_index[en_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True,
            )
            transformer.eval()
            # Prepare encoder input, mask and generate output hidden states
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(corpus, add_special_tokens=False)
            )
            src_padding_mask = encoder_input != transformer.padding_idx
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)
            # Prepare decoder input, mask and start decoding
            decoder_input = torch.IntTensor(
                [[transformer.bos_idx], [transformer.bos_idx]]
            )
            future_mask = construct_future_mask(seq_len=1)
            for _ in range(transformer.max_decoding_length):
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                # Take the argmax over the softmax of the last token
                # to obtain the next token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)

                # Append the prediction to the already
                # decoded tokens and construct the new mask
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

                self.assertEqual(
                    decoder_input.shape, (2, transformer.max_decoding_length + 1)
                )
                # See test_one_layer_transformer_decoder_inference
                # in decoder.py for more inforemation. With num_layers=1,
                # this will be true.
                self.assertEqual(torch.all(decoder_input == transformer.bos_idx), False)


if __name__ == "__main__":
    unittest.main()

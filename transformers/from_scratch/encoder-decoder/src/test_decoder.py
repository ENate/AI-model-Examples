"""The test file for decoder."""

import random
import unittest
import numpy as np
import torch

from .utils import construct_future_mask

from .decoder import TransformerDecoder


class TestTransformerDecoder(unittest.TestCase):
    """Defines a given test case.

    Args:
        unittest(object): represents a given test object.
    """

    def test_one_layer_transformer_decoder_inference(self):
        """
        Test two forward passes, simulating greedy decoding
        test_one_layer_transformer_decoder_inference
        steps.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        with torch.no_grad():
            batch_size = 2
            src_seq_len = 10
            hidden_dim = 512
            vocab_size = 2000
            num_layers = 1
            num_heads = 8

            # Prepare fake encoder hidden states and padding masks
            encoder_output = torch.randn((batch_size, src_seq_len, hidden_dim))
            src_padding_mask = torch.BoolTensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )
            # Initialize the decoder, perform xavier init and set to
            # evaluation mode
            decoder = TransformerDecoder(
                embedding=torch.nn.Embedding(vocab_size, hidden_dim),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_p=0.1,
                vocab_size=vocab_size,
                tie_output_to_embedding=True,
            )
            decoder._reset_parameters()
            decoder.eval()
            # Prepare decider input, mask, perform a decoding step,
            # take the argmax over the softmax of the last token
            bos_token_id = 1
            decoder_input = torch.IntTensor([[bos_token_id], [bos_token_id]])
            future_mask = None
            for i in range(3):
                decoder_output = decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze_(1)
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])
                self.assertEqual(decoder_output.shape, (batch_size, i + 1, vocab_size))
                # Check: softmax entropy should not be zero
                self.assertEqual(torch.any(decoder_output == 1), False)

                """
                With only one decoder layer the predicted tokens will always be the input token ids. This happens
                only when the final linear transformation is tied to the (transpose of) the embedding matrix.
                This is because the input embedding is barely transformed due to residual connections. This results in
                the highest dot product between its final "contextualized" embedding and the original embedding vector
                in the pre-softmax weight matrix (i.e. embedding matrix) - because they are still very similar.
                This can be avoided by 1) scaling up the memory states - probably because this adds sufficient random
                noise through cross-attention to the contextualised embedding to divergence from the input embedding.
                2) increasing the number of layers - again adding more and more "noise" or 3) removing the last
                residual connection after the feed forward layers. In practice, however, this is not an issue. Training
                will take care of it.
                """

                self.assertEqual(torch.all(decoder_input == bos_token_id), False)

    def test_multi_layer_transformer_decoder_inference(self):
        """
        Test two forward passes, simulating two inference decoding steps
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            batch_size = 2
            src_seq_len = 10
            hidden_dim = 512
            vocab_size = 2000

            # Prepare fake encoder hidden states and padding masks
            encoder_output = torch.randn((batch_size, src_seq_len, hidden_dim))
            src_padding_mask = torch.BoolTensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )

            # Initialize the decoder, perform xavier init and set to evaluation mode
            decoder = TransformerDecoder(
                embedding=torch.nn.Embedding(vocab_size, hidden_dim),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                dropout_p=0.1,
                vocab_size=vocab_size,
                tie_output_to_embedding=False,
            )
            decoder._reset_parameters()
            decoder.eval()

            # Prepare decoder input, mask, perform a decoding step, take the argmax over the softmax of the last token
            bos_token_id = 10
            # and iteratively feed the input+prediction back in.
            decoder_input = torch.IntTensor([[bos_token_id], [bos_token_id]])
            future_mask = None
            for i in range(3):
                decoder_output = decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

                self.assertEqual(decoder_output.shape, (batch_size, i + 1, vocab_size))
                # softmax entropy should not be 0
                self.assertEqual(torch.any(decoder_output == 1), False)
                self.assertEqual(torch.all(decoder_input == bos_token_id), False)


if __name__ == "__main__":
    unittest.main()

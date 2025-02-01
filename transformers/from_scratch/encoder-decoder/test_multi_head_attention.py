"""An implementation to test the multi-headed attention."""

import torch
import unittest
from src.multi_head_attention import MultiHeadAttention
from src.utils import construct_future_mask


class TestMultiHeadAttention(unittest.TestCase):
    def test_scaled_dot_product(self):
        mha = MultiHeadAttention(512, 8)
        q = torch.randn(4, 8, 10, 512)
        k = torch.randn(4, 8, 10, 512)
        v = torch.randn(4, 8, 10, 512)

        values, attention_scores = mha.scaled_dot_product(q, k, v)
        self.assertEqual(values.shape, (4, 8, 10, 512))
        self.assertEqual(attention_scores.shape, (4, 8, 10, 10))

        # Each attention distribution should sum up to one
        expected = torch.Tensor([1.0]).repeat((4, 8, 10))
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        self.assertEqual(torch.any(torch.isnan(values)), False)
        self.assertEqual(True in torch.isnan(attention_scores), False)

    def test_scalar_dot_product(self):
        mha = MultiHeadAttention(hidden_dim=512, num_heads=8)
        q = torch.randn(2, 8, 10, 512, dtype=torch.float)
        k = torch.randn(2, 8, 10, 512, dtype=torch.float)
        v = torch.randn(2, 8, 10, 512, dtype=torch.float)

        mask = torch.BoolTensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )
        _, attention_scores = mha.scaled_dot_product(q, k, v, src_padding_mask=mask)
        self.assertEqual(torch.any(torch.isnan(attention_scores)), False)
        # For the first sequence, we expect the last two (8-10) attention scores
        # for every attention distribution
        # For every head to be exactly zero due to the mask defined above.
        # The rest should be strictly non-zero.

        self.assertEqual(torch.all(attention_scores[0, :, :, 8:] == 0), True)
        self.assertEqual(torch.any(attention_scores[0, :, :, :8] == 0), False)
        # Check if all attention distribution will sum up to 1! (i.e all values after summing
        # should be 1!).
        expected = torch.Tensor([1.0]).repeat([2, 8, 10])
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        # For some second sequence in the batch, all attention scores
        # should be non zero because the mask is all ones
        self.assertEqual(torch.any(attention_scores[1] == 0), False)

    def test_mha_self_attention_forward(self):
        mha = MultiHeadAttention(512, 8)
        x = torch.randn(4, 10, 512, dtype=torch.float)
        output = mha.forward(x)
        # Check dimension of attention matrix
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_cross_attention_projection(self):
        mhs = MultiHeadAttention(512, 8)
        decoder_hidden_states = torch.randn(4, 2, 512, dtype=torch.float)
        encoder_hidden_states = torch.randn(4, 2, 512, dtype=torch.float)
        output = mhs.forward(
            x=decoder_hidden_states, encoder_hidden_states=encoder_hidden_states
        )
        self.assertEqual(output.shape, (4, 2, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_future_masking(self):
        batch_size, num_heads, seq_len = 2, 2, 3  # Add 2 , 3
        logits = torch.randn(
            batch_size, num_heads, num_heads, seq_len, seq_len, dtype=torch.float
        )
        future_mask = construct_future_mask(seq_len)
        self.assertEqual(future_mask.shape, (3, 3))
        masked_logits = MultiHeadAttention(512, num_heads=num_heads).mask_logits(
            logits, future_mask=future_mask
        )
        torch.testing.assert_close(
            torch.isinf(masked_logits) == 0, torch.BoolTensor([[]])
        )


if __name__ == "__main__":
    unittest.main()

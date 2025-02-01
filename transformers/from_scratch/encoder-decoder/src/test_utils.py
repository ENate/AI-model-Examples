"""To test the utils py file."""

import torch
import unittest

from vocabulary import Vocabulary
from utils import construct_batches, construct_future_mask


class TestUtils(unittest.TestCase):
    def test_construct_future_mask(self):
        mask = construct_future_mask(3)
        torch.testing.assert_close(
            mask,
            torch.BoolTensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
        )

    def test_construct_future_mask_first_decoding_step(self):
        mask = construct_future_mask(1)
        torch.testing.assert_close(mask, torch.BoolTensor([[True]]))

    def test_construct_batches(self):
        corpus = [
            {"en": "This is an english sentence.", "nl": "Dit is een Nederlandse zin."},
            {"en": "The weather is nice today.", "nl": "Het is lekker weer vandaag."},
            {
                "en": "Yesterday I drove to a city called Amsterdam in my brand new car.",
                "nl": "Ik reed gisteren in mijn gloednieuwe auto naar Amsterdam.",
            },
            {
                "en": "You can pick up your laptop at noon tomorrow.",
                "nl": "Je kunt je laptop morgenmiddag komen ophalen.",
            },
        ]
        en_sentences, nl_sentences = (
            [d["en"] for d in corpus],
            [d["nl"] for d in corpus],
        )
        vocab = Vocabulary(en_sentences + nl_sentences)
        batches, masks = construct_batches(
            corpus, vocab, batch_size=2, src_lang_key="en", tgt_lang_key="nl"
        )
        torch.testing.assert_close(
            batches["src"],
            [
                torch.IntTensor(
                    [[0, 3, 4, 5, 6, 7, 8, 1], [0, 9, 10, 4, 11, 12, 8, 1]]
                ),
                torch.IntTensor(
                    [
                        [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 1],
                        [0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 8, 1, 2, 2, 2, 2],
                    ]
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

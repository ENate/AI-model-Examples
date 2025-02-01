import unittest
import random
from random import choices

import numpy as np
import torch
from torch import nn

from src.lr_scheduler import NoamOpt
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from src.utils import construct_batches
from src.train import train


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def test_copy_task(self):
        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if device.type == "cpu":
            print("This unit test was not run because it requires a GPU")
            return

        # Hyperparameters
        synthetic_corpus_size = 600
        batch_size = 60
        n_epochs = 200
        n_tokens_in_batch = 10

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        corpus = ["These are the tokens that will end up in our vocabulary"]
        vocab = Vocabulary(corpus)
        vocab_size = len(
            list(vocab.token_2_index.keys())
        )  # 14 tokens including bos, eos and pad
        valid_tokens = list(vocab.token_2_index.keys())[3:]
        corpus += [
            " ".join(choices(valid_tokens, k=n_tokens_in_batch))
            for _ in range(synthetic_corpus_size)
        ]

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]
        batches, masks = construct_batches(
            corpus,
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=25,
            vocab_size=vocab_size,
            padding_idx=vocab.token_2_index[vocab.PAD],
            bos_idx=vocab.token_2_index[vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamOpt(
            transformer.hidden_dim,
            factor=1,
            warmup=400,
            optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
        )
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()

"""The train.py implementation."""

import torch
from torch import nn
from typing import List, Dict, Any


def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0

    for e in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgr"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)
            # Perform one decoder forward pass to obtain *all* next-token predictions for every 
            # index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )
            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]
            # Set pad tokens in the target to -100 so they don't incur a loss
            # tgt_batch[tgt_batch == transformer.padding_idx] = -100
            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )
            # Rough estimate of per-token accuracy in the current training batch
            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / torch.numel(tgt_batch)

            if num_iters % 100 == 0:
                print(
                    f"epoch: {e}, num_iters:  {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )
            # update parameters
            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_loss, batch_accuracy

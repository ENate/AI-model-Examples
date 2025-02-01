from typing import Dict, List, Tuple, Optional
import torch
from vocabulary import Vocabulary


def construct_future_mask(seq_len: int):
    """
    Construct a binary mask that contains 1's for all
    valid connections and 0's for all outgoing future connections.
    This mask will be applied to the attention logits in
    decoder self-attention such that all logits with a 0 mask
    are set to -inf.

    :param seq_len: length of the input sequence
    :return: (seq_len,seq_len) mask
    """
    subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
    return subsequent_mask == 0


def construct_batches(
    corpus: List[Dict[str, str]],
    vocab: Vocabulary,
    batch_size: int,
    src_lang_key: str,
    tgt_lang_key: str,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Constructs batches given a corpus.

    :param corpus: The input corpus is a list of aligned source and target
    sequences, packed in a dictionary.
    :param vocab: The vocabulary object.
    :param batch_size: The number of sequences in a batch

    :param src_lang_key: The source language key is a
    string that the source sequences are keyed under. E.g. "en"
    :param tgt_lang_key: The target language key is
    a string that the target sequences are keyed under. E.g. "nl"
    :param device: whether or not to move tensors to gpu
    :return: A tuple containing two dictionaries. The
    first represents the batches, the second the attention masks.
    """
    pad_token_id = vocab.token_2_index[vocab.PAD]
    batches: Dict[str, List] = {"src": [], "tgt": []}
    masks: Dict[str, List] = {"src": [], "tgt": []}
    for i in range(0, len(corpus), batch_size):
        src_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[src_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )
        tgt_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[tgt_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )

        src_padding_mask = src_batch != pad_token_id
        future_mask = construct_future_mask(tgt_batch.shape[-1])

        # Move tensors to gpu; if available
        if device is not None:
            src_batch = src_batch.to(device)  # type: ignore
            tgt_batch = tgt_batch.to(device)  # type: ignore
            src_padding_mask = src_padding_mask.to(device)
            future_mask = future_mask.to(device)
        batches["src"].append(src_batch)
        batches["tgt"].append(tgt_batch)
        masks["src"].append(src_padding_mask)
        masks["tgt"].append(future_mask)
    return batches, masks

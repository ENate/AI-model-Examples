"""Defines the vabulary or sentence tokenizer."""
import re
from typing import Optional, List


class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]):
        """Initializes the parameters."""
        self.token_2_index = { self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index_2_token = {v: k for k, v in self.token_2_index.items()}
        if not list_of_sentences:
            return
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))
    
    def add_tokens(self, tokens: List[str]) -> None:
        """Adds tokens to vocab
        :param tokens - list of tokens
        """
        for token in tokens:
            if token not in self.token_2_index:
                i = len(self.token_2_index.items())
                self.token_2_index[token] = i
                self.index_2_token[i] = token
    
    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """Adds tokens to sentences by splits and punctuations

        Args:
            sentence (str): str of sentences
            add_special_tokens (bool, optional): checks whether to add BIOS, etc.
            defaults to True.

        Returns:
            List[str]: returns a list of tokens
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens
    
    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """Converts a string to a list of token indices given the vocabulary.

        Args:
            sentence (str): a string representation of a sentence.
            add_special_tokens (bool, optional): Whether to add BOS and EOS.
            Defaults to True.

        Returns:
            List[str]: returns list of token indices.
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token_2_index[token] for token in tokens]
    
    def batch_encode(self, sentences: List[str], padding = True, 
    add_special_tokens: bool = False) -> List [List[int]]:
        """Convert a list of string sentences to nested list of token indices. 
        Optionally adds padding & bos+eos tokens

        Args:
            sentence (List[str]): A list of sentences to be encoded into a batch
            padding (bool, optional): Boolean allows for padding up to the longest
            sentence.
            add_special_tokens (bool, optional): Boolean that allows for adding a 
            BOS and EOS token to each sentence in the batch
            Defaults to True.

        Returns:
            List [List[int]]: nested list of tokenized sequences
        """
        token_sequences = [
            self.encode(sentence, add_special_tokens) for sentence in sentences]
        if padding:
            max_length = max([len(tokens) for tokens in token_sequences])
            token_sequences = [
                s + ((max_length - len(s)) * [self.token_2_index[self.PAD]])
                for s in token_sequences
            ]
        return token_sequences


    


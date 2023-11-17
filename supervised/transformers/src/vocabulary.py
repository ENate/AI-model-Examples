import re
# check why we need this lin
from typing import List, Optional


class Vocabulary:
    """Generates a list of tokens from imput string """
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]) -> None:
        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
        # check whethere there is a list of sentences?
        if not list_of_sentences:
            # then return
            return
        # But if there is a list of sentences, call add token method,
        # tokenizes and substitute the values for BOS, EAS, PADS?
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Adds token to the vocabulary
        :param tokens - represents a tokenized sentence
        :return None
        """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(
            self,
            sentence: str,
            add_boolean_tokens: bool = True) -> List[str]:
        """
        Splits all tokens and punctuations. Adds BOS and EOS optionally.
        :param: sentence
        :param: add_boolean_tokens
        :return list of long tokens
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_boolean_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(
            self, sentence: str,
            add_special_tokens: bool = True) -> List[str]:
        pass

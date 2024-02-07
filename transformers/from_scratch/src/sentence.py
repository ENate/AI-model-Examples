import re
from typing import List, Optional


class Sentence:
    """Generate a list of tokens from a list of input sentences """
    BOS = "BOS"  # Beginning of sentence
    EOS = "EOS"  # End of Sentence
    PAD = "PAD"
    
    def __init__(self, sentence_list: Optional[List[str]] = None):
        self.token_2_index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index_to_token = {v: k for k, v in self.token_2_index.items()}
        # Check whether there is a list of input sentences
        if not sentence_list:
            return  # then return to initial state
        
        # Call token method if there is a list of sentences
        # Tokenize and substitute BOS, EAS and PAD
        for sentence in sentence_list:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self, param):
        pass
        
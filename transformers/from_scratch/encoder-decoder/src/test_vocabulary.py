"""A test implementation for the vocabulary script."""
import unittest
from vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    # Test the tokenize function.
    def test_tokenize(self):
        input_sentence = "Hey, there I am here!"
        tokened_output = Vocabulary([]).tokenize(input_sentence)
        print(tokened_output)
        self.assertEqual(["BOS", "Hey", ",", "there", "I", "am", "here", "!", "EOS"], tokened_output)

    def test_initalize_vocab(self):
        input_sentence = ["May the force be with you."]
        vocab = Vocabulary(input_sentence)
        expected = {"BOS": 0, "EOS":1, "PAD":2, "May": 3, "the": 4, "force": 5, "be": 6, "with": 7, "you": 8, ".": 9}
        self.assertEqual(vocab.token_2_index, expected)
    
    def test_encode(self):
        input_sentence = ["May the force be with you."]
        vocab = Vocabulary(input_sentence)
        output = vocab.encode(input_sentence[0])
        print(output)
        self.assertEqual(output, [0, 3, 4, 5, 6, 7, 8, 9, 1])
    
    def test_encode_no_special_tokens(self):
        input_sentence = ["May the force be with you."]
        vocab = Vocabulary(input_sentence)
        output = vocab.encode(input_sentence[0], add_special_tokens=False)
        self.assertEqual(output, [3, 4, 5, 6, 7, 8, 9])
    
    def test_batch_encode(self):
        input_sentences = [
            "Round the rough and rugged road",
            "The rugged rascal ruddely ran",
            "Two tiny timid toads trying to troad to tarrytown"
        ]
        vocab = Vocabulary(input_sentences)
        output = vocab.batch_encode(input_sentences, add_special_tokens=False)
        print(output)
        input_vec = [
            [3, 4, 5, 6, 7, 8, 2, 2, 2], 
            [9, 7, 10, 11, 12, 2, 2, 2, 2], 
            [13, 14, 15, 16, 17, 18, 19, 18, 20]]
        self.assertEqual(output, input_vec)


if __name__ == "__main__":
    unittest.main()


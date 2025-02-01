"""A test implementation for the positional_encoding."""
import torch
import unittest
from positional_encoding import SinusoidalEncoding

class TestSinusoidalEncoding(unittest.TestCase):

    def test_create_embedding(self):
        batch = 1
        dim = 8
        len = 3
        x = torch.zeros(batch, len, dim)
        encoding = SinusoidalEncoding(dim).forward(x)
        expected = torch.Tensor([
            [
                [
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                ],
                [
                    8.4147e-01,
                    5.4030e-01,
                    9.9833e-02,
                    9.9500e-01,
                    9.9998e-03,
                    9.9995e-01,
                    1.0000e-03,
                    1.0000e00,
                ],
                [
                    9.0930e-01,
                    -4.1615e-01,
                    1.9867e-01,
                    9.8007e-01,
                    1.9999e-02,
                    9.9980e-01,
                    2.0000e-03,
                    1.0000e00,
                ]
            ]
        ])
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)
    
    def test_create_embedding_multi_batch(self):
        batch = 2
        dim = 8
        len = 3
        x = torch.zeros(batch, len, dim)
        encoding = SinusoidalEncoding(dim).forward(x)
        expected = torch.Tensor(
            [
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
            ]
        )
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)



if __name__ == "__main__":
    unittest.main()

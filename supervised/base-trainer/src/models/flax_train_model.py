"""An example of FLAX model """
import jax
import jax.numpy as jp

from flax import linen as nn


class CNN(nn.Module):
    """Define CNN model by stacking layers """
    @nn.compact
    def __call__(self, x):
        """Layers  """
        x = nn.Conv(features = 32, kernel_size = (3, 3))(x)
        x = nn.relu(x)
        # Add pooling and strides
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

    def print_cnn(self, vec_structure, prng_key):
        """ Print or visualize CNN architecture """


if __name__ == '__main__':
    cnn = CNN()
    print(cnn.tabulate(jax.random.PRNGKey(0), jp.ones((1, 28, 28, 1))))

""" Import data and prepare features """
import tensorflow as tf
import tensorflow_datasets as tfds


class Features:
    """ Main features class """

    def __init__(self, x_num, y_num) -> None:
        """ Initialize the main class variables """
        self.x_num = x_num
        self.y_num = y_num

    def build_features(self, num):
        """ prepare data set for training """
        pass

    def flax_model_example(self, num_epochs, batch_size):
        """ Example to demonstrate flax """
        """Load MINST Dataset in the memory"""
        train_data = tfds.load('minst', split='train')
        test_data = tfds.load('minst', split='test')

        train_ds = train_data.map(lambda sample: {'image': tf.cast(sample['image'],
                                                                   tf.float32) / 255.,
                                                  'label': sample['label']})  # normalize train set
        test_ds = test_data.map(lambda sample: {'image': tf.cast(sample['image'],
                                                                 tf.float32) / 255.,
                                                'label': sample['label']})  # normalize test set
        # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
        train_ds = train_ds.repeat(num_epochs).shuffle(1024)
        # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
        train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
        # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
        test_ds = test_ds.shuffle(1024)
        # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
        test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
        return train_ds, test_ds

    def check_dataset_feats(self):
        """ final features preparation prior to training """

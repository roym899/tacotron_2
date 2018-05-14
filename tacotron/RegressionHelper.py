import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
import numpy as np


class RegressionHelper(Helper):
    """Interface for implementing sampling in seq2seq decoders.
    Helper instances are used by `BasicDecoder`.
    """

    def __init__(self, batch_size, frequency_bins, max_output_length):
        self.frequency_bins = frequency_bins
        self._batch_size = batch_size
        self.max_output_length = max_output_length

    @property
    def batch_size(self):
        """Batch size of tensor returned by `sample`.
        Returns a scalar int32 tensor.
        """
        return self._batch_size

    @property
    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.
        Returns a `TensorShape`.
        """
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.
        Returns a DType.
        """
        return np.int32

    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        return (tf.tile([False], [self._batch_size]), tf.zeros([self._batch_size, self.frequency_bins]))

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        stop = tf.cond(time >= self.max_output_length-1, lambda: True, lambda: False)
        return (stop, outputs, state)

import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
import numpy as np

class RegressionHelper(Helper):
    """Interface for implementing sampling in seq2seq decoders.
    Helper instances are used by `BasicDecoder`.
    """

    def __init__(self, batch_size, frequency_bins, max_output_length, pre_net=None):
        """

        :param batch_size:
        :param frequency_bins:
        :param max_output_length:
        :param pre_net: list of the pre net layers, will be applied on the output in the order of the list
        """
        self.frequency_bins = frequency_bins
        self._batch_size = batch_size
        self.max_output_length = max_output_length
        self.pre_net = pre_net

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
        if self.pre_net is None:
            return (tf.tile([False], [self._batch_size]), tf.zeros([self._batch_size, self.frequency_bins]))
        else:
            return (tf.tile([False], [self._batch_size]), tf.zeros([self._batch_size, self.pre_net[0].units]))

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        if self.pre_net is not None:
            out1 = self.pre_net[0](outputs)
            out2 = self.pre_net[1](out1)
        else:
            out2 = outputs
        stop = tf.cond(time >= self.max_output_length-1, lambda: True, lambda: False)
        return (stop, out2, state)

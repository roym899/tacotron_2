import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


class RegressionHelper(Helper):
    """Interface for implementing sampling in seq2seq decoders.
    Helper instances are used by `BasicDecoder`.
    """

    def __init__(self, max_output_length):
        self.max_output_length = max_output_length

    def batch_size(self):
        """Batch size of tensor returned by `sample`.
        Returns a scalar int32 tensor.
        """
        return 1

    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.
        Returns a `TensorShape`.
        """
        return [1]

    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.
        Returns a DType.
        """
        return tf.int32

    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        return (False, 0)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        return [0]

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        stop = False
        if time > self.max_output_length:
            stop = True
        return (stop, outputs, state)

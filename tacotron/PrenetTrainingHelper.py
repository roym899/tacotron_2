import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from tensorflow.contrib.seq2seq.python.ops import helper
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


class PrenetTrainingHelper(Helper):
  """A helper for use during training.  Only reads inputs.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, pre_net, time_major=False, name=None):
    """Initializer.
    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "PrenetTrainingHelper", [inputs, sequence_length]):
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = nest.map_structure(helper._transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(helper._unstack_ta, inputs)
      self._sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

      self._batch_size = array_ops.size(sequence_length)
      self.pre_net = pre_net

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.int32

  def initialize(self, name=None):
    with ops.name_scope(name, "TrainingHelperInitialize"):
      return (tf.tile([False], [self._batch_size]), tf.zeros([self._batch_size, self.pre_net[0].units]))
      # finished = math_ops.equal(0, self._sequence_length)
      # all_finished = math_ops.reduce_all(finished)
      # next_inputs = control_flow_ops.cond(
      #     all_finished, lambda: self._zero_inputs,
      #     lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
      # return (finished, next_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with ops.name_scope(name, "TrainingHelperNextInputs",
                        [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(next_time)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(read_from_ta, self._input_tas))

      out1 = self.pre_net[0](next_inputs)
      out2 = self.pre_net[1](out1)
      return (finished, out2, state)
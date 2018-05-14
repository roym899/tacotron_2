# define the basic vocabulary
VOCAB = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-?!.,;:\'"

import os
import numpy as np
import tensorflow as tf
from wavenet import wavenet

def text_to_sequence(text, length):
  sequence = []
  for _ in range(length):
    sequence.append(0)
  for i, c in enumerate(text):
    idx = VOCAB.find(c)
    assert idx != -1, "character {} not found".format(c)
    sequence[i] = idx

  return sequence


def process_data(abs_path, hparams):
  with open(os.path.join(abs_path, '..', 'dataset', 'test.data')) as f:
    data = {}
    lines = f.readlines()
    num = len(lines)
    input_sequence = np.zeros(shape=(num, hparams['max_sentence_length']))
    target_spectogram = np.zeros(shape=(num, 500, 128))

    for index, line in enumerate(lines):
      # line structure: ( wav_name "text" )
      parts = line.strip().split('"')
      wav_name = parts[0].strip('( ')
      text = parts[1]

      # convert audio to spectogram
      audio = wavenet.load_audio(os.path.join(abs_path, '..', 'dataset', 'wavn', wav_name + '.wav'), expected_samplerate=16000)
      spectogram = wavenet.calculate_stft(audio, hparams['fftsize'], hparams['hops'])
      spectogram = np.pad(spectogram,
                          ((0, hparams['max_output_length'] - spectogram.shape[0]), (0, 0)),
                          'constant')
      spectogram = wavenet.calculate_mag(spectogram)
      #spectogram = np.reshape(spectogram, (1, np.prod(np.shape(spectogram))))
      target_spectogram[index] = spectogram

      # convert text to sequence
      sequence = text_to_sequence(text, hparams['max_sentence_length'])
      input_sequence[index] = sequence

    np.save('sequence.npy', input_sequence)
    np.save('spectogram.npy', target_spectogram)


def load_dataset():
  sequence = np.load("sequence.npy")
  spectogram = np.load("spectogram.npy")

  # check the dimension
  assert sequence.shape[0] == spectogram.shape[0]

  return sequence, spectogram

  # sequence_placeholder = tf.placeholder(sequence.dtype, sequence.shape)
  # spectogram_placeholder = tf.placeholder(spectogram.dtype, spectogram.shape)
  #
  # dataset = tf.data.Dataset.from_tensor_slices((sequence_placeholder, spectogram_placeholder))
  #
  # iterator = dataset.make_initializable_iterator()
  #
  # with tf.Session as sess:
  #   sess.run(iterator.initializer, feed_dict={sequence_placeholder: sequence,
  #                                           spectogram_placeholder: spectogram})
  #
  # dataset1 = tf.data.Dataset.from_sparse_tensor_slices(sequence)
  # dataset2 = tf.data.Dataset.from_sparse_tensor_slices(spectogram)
  # dataset = tf.data.Dataset.zip((dataset1, dataset2))
  # batched_dataset = dataset.batch(4)
  # iterator = batched_dataset.make_one_shot_iterator()
  # next_element = iterator.get_next()
  #
  # with tf.Session as sess:
  #   print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
  #   print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
  #   print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])



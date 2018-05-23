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


def process_data(dataset_path, hparams, max_dataset_size,processed_path):
  print("Processing data...")
  with open(os.path.join(dataset_path, 'prompts.data')) as f:
    lines = f.readlines()
    num = len(lines)

    files = num // max_dataset_size + 1
    file_counter = 0
    processed = 0

    skipped_sentence = 0
    skipped_output = 0

    input_sequence = np.zeros(shape=(max_dataset_size, hparams['max_sentence_length']))
    input_sequence_length = np.zeros(shape=(max_dataset_size))
    target_spectogram = np.zeros(shape=(max_dataset_size, hparams['max_output_length'], hparams['frequency_bins']))

    filtered_sentences = []

    for index, line in enumerate(lines):
      # line structure: ( wav_name "text" )
      parts = line.strip().split('"')
      wav_name = parts[0].strip('( ')
      text = parts[1]

      if len(text) > hparams['max_sentence_length']:
        skipped_sentence += 1
        continue

      # convert audio to spectogram
      audio = wavenet.load_audio(os.path.join(dataset_path, 'wavn', wav_name + '.wav'), expected_samplerate=44100)
      spectogram = wavenet.calculate_stft(audio, hparams['fftsize'], hparams['hops'])
      if hparams['max_output_length'] - spectogram.shape[0] < 0:
        skipped_output += 1
        continue
        # spectogram = spectogram[0:hparams['max_output_length'], :]
      else:
        spectogram = np.pad(spectogram,
                            ((0, hparams['max_output_length'] - spectogram.shape[0]), (0, 0)),
                            'constant')
      spectogram = wavenet.calculate_mag(spectogram, hparams['frequency_bins'])
      #spectogram = np.reshape(spectogram, (1, np.prod(np.shape(spectogram))))
      target_spectogram[processed] = spectogram

      # convert text to sequence
      input_sequence_length[processed] = len(text)
      sequence = text_to_sequence(text, hparams['max_sentence_length'])
      input_sequence[processed] = sequence

      filtered_sentences.append(text)

      print("Processing data... {}/{} ({} skipped due to length, {} skipped due to out length)".format(index, len(lines), skipped_sentence, skipped_output))
      processed += 1
      if processed >= max_dataset_size:
        if file_counter+max_dataset_size > num:
          end = num
        else:
          end = file_counter+1000
        np.save(os.path.join(processed_path, 'sequence_length_{}.npy'.format(file_counter)), input_sequence_length)
        np.save(os.path.join(processed_path, 'sequence_{}.npy'.format(file_counter)), input_sequence)
        np.save(os.path.join(processed_path, 'spectogram_{}.npy'.format(file_counter)), target_spectogram)

        if num-index > max_dataset_size:
          next_size = max_dataset_size
        else:
          next_size = num-index
        input_sequence = np.zeros(shape=(next_size, hparams['max_sentence_length']))
        target_spectogram = np.zeros(shape=(next_size, hparams['max_output_length'], hparams['frequency_bins']))
        processed = 0
        file_counter += 1

  with open(os.path.join(dataset_path, 'filtered_prompts.data'), "w") as file:
    for sentence in filtered_sentences:
      file.write("%s\n" % sentence)


def load_dataset(dataset_path, id):
  sequence = np.load(os.path.join(dataset_path, 'sequence_{}.npy'.format(id)))
  spectogram = np.load(os.path.join(dataset_path, 'spectogram_{}.npy'.format(id)))
  sequence_length = np.load(os.path.join(dataset_path, 'sequence_length_{}.npy'.format(id)))

  # check the dimension
  assert sequence.shape[0] == spectogram.shape[0]

  return sequence, sequence_length, spectogram

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



# define the basic vocabulary
VOCAB = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-?!.,;:\'"

import os
import numpy as np
import pandas as pd
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


def load_data(abs_path, hparams):
  with open(os.path.join(abs_path, '..', 'dataset', 'prompts.data')) as f:
    data = {}
    lines = f.readlines()
    num = len(lines)
    #input_sequence = np.zeros(shape=(num, hparams['max_sentence_length']))
    #target_spectogram = np.zeros(shape=(num, hparams['max_output_length'], hparams['frequency_bins']))

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
      spectogram = np.reshape(spectogram, (hparams['max_output_length'] * hparams['fftsize'], 1))

      np.savetxt('spectogram.npy', spectogram, delimiter=',')

      # convert text to sequence
      sequence = text_to_sequence(text, hparams['max_sentence_length'])

      seq_frame = pd.DataFrame({'sequence': sequence})
      seq_frame.to_csv("sequence.csv", index=False, sep=',')




# Tacotron 2 Main File
import dataset
from tacotron.model import TTS
import tacotron
import os
import numpy as np
import local_paths

# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data


# TEST PARAMS
# hparams = {}
# hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
# hparams['embedding_size'] = 100
# hparams['max_sentence_length'] = 80
# hparams['basic_encoder_lstm_cells'] = 512
# hparams['fftsize'] = 2048
# hparams['hops'] = 2048 // 8
# hparams['frequency_bins'] = 128
# hparams['max_output_length'] = 10
# hparams['max_gradient_norm'] = 5
# hparams['learning_rate'] = 10e-3


# BATCH PARAMS
hparams = {}
hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
hparams['embedding_size'] = 512
hparams['max_sentence_length'] = 150
hparams['basic_encoder_lstm_cells'] = 512
hparams['fftsize'] = 2048
hparams['hops'] = 2048 // 8
hparams['frequency_bins'] = 128
hparams['max_output_length'] = 700
hparams['learning_rate'] = 10e-3
hparams['batch_size'] = 64

# First check if dataset is available, otherwise cancel
assert os.path.exists(local_paths.DATASET_PATH + "prompts.data"), "Missing text dataset!"
assert os.path.exists(local_paths.DATASET_PATH + "wavn"), "Missing audio dataset!"
# Check if it has already been processed
if not os.path.exists(local_paths.DATASET_PATH + "sequence.npy") or\
        not os.path.exists(local_paths.DATASET_PATH + "spectogram.npy") or\
        np.shape(np.load(local_paths.DATASET_PATH + "sequence.npy"))[1] != hparams['max_sentence_length'] or\
        np.shape(np.load(local_paths.DATASET_PATH + "spectogram.npy"))[1:3] != (hparams['max_output_length'], hparams['frequency_bins']):
    # process the data
    tacotron.utils.process_data(local_paths.DATASET_PATH, hparams)
# # Load dataset
training_sequences, training_spectograms = tacotron.utils.load_dataset(local_paths.DATASET_PATH)

improved_tacotron_2_model = TTS(hparams, "basic")

# limit the dataset such that the model still runs
training_sequences = training_sequences[0:500,:]
training_spectograms = training_spectograms[0:500,:]

# improved_tacotron_2_model.train_test()
improved_tacotron_2_model.train(training_sequences, training_spectograms)

# TODO: save model

improved_tacotron_2_model.predict("This course is fun.")


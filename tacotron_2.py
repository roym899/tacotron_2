# Tacotron 2 Main File
import dataset
from tacotron.model import TTS
import tacotron
import os
import numpy as np

# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data
DATASET_PATH = "../dataset/"


hparams = {}
hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
hparams['embedding_size'] = 512
hparams['max_sentence_length'] = 400
hparams['basic_encoder_lstm_cells'] = 512
hparams['fftsize'] = 2048
hparams['hops'] = 2048 // 8
hparams['frequency_bins'] = 128
hparams['max_output_length'] = 120
hparams['max_gradient_norm'] = 5
hparams['learning_rate'] = 10e-5
hparams['number_conv_layers_encoder'] = 3
hparams['is_Training'] = True

# # First check if dataset is available, otherwise cancel
# assert os.path.exists(DATASET_PATH + "prompts.data"), "Missing text dataset!"
# assert os.path.exists(DATASET_PATH + "wavn"), "Missing audio dataset!"
# # Check if it has already been processed
# if not os.path.exists(DATASET_PATH + "sequence.npy") or\
#         not os.path.exists(DATASET_PATH + "spectogram.npy") or\
#         np.shape(np.load(DATASET_PATH + "sequence.npy"))[1] != hparams['max_sentence_length'] or\
#         np.shape(np.load(DATASET_PATH + "spectogram.npy"))[1:3] != (hparams['max_output_length'], hparams['frequency_bins']):
#     # process the data
#     tacotron.utils.process_data(hparams)
# # Load dataset
# training_sequences, training_spectograms = tacotron.utils.load_dataset()



improved_tacotron_2_model = TTS(hparams, "convolutional")

improved_tacotron_2_model.train_test()
# improved_tacotron_2_model.train(training_sequences, training_spectograms)

# TODO: save model

improved_tacotron_2_model.predict("This course is fun.")


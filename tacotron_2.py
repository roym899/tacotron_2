# Tacotron 2 Main File
import dataset
from tacotron.model import TTS, TTS_Mode
import tacotron
import os
import numpy as np
import local_paths

# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data



### Setup the network mode
# Examples (all BASIC components are always enabled/cannot be disabled)
# Just add convolution: mode = TTS_MODE.CONVOLUTIONAL
# Combine Convoluition and 2 Layer LSTM: mode = TTS_MODE.CONVOLUTIONAL | TTS_MODE.TWO_LSTM_DECODER
mode = TTS_Mode.BIDIRECTIONAL_LSTM_ENCODER
# mode = TTS_Mode.BASIC
# activate/deactivate test mode, will skip the dataset loading
test = True

if test:
    # TEST PARAMS
    hparams = {}
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 100
    hparams['max_sentence_length'] = 80
    hparams['basic_lstm_cells'] = 512
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 8
    hparams['frequency_bins'] = 128
    hparams['max_output_length'] = 120
    hparams['max_gradient_norm'] = 5
    hparams['learning_rate'] = 10e-3
    hparams['batch_size'] = 64
    hparams['number_conv_layers_encoder'] = 3
    hparams['is_Training'] = True
else:
    # BATCH PARAMS
    hparams = {}
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 512
    hparams['max_sentence_length'] = 150
    hparams['basic_encoder_lstm_cells'] = 512
    hparams['basic_decoder_lstm_cells'] = 256
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 8
    hparams['frequency_bins'] = 128
    hparams['max_output_length'] = 700
    hparams['learning_rate'] = 10e-3
    hparams['batch_size'] = 64
    hparams['number_conv_layers_encoder'] = 3
    hparams['is_Training'] = True



improved_tacotron_2_model = TTS(hparams, mode)

if test is False:
    # First check if dataset is available, otherwise cancel
    assert os.path.exists(local_paths.DATASET_PATH + "prompts.data"), "Missing text dataset!"
    assert os.path.exists(local_paths.DATASET_PATH + "wavn"), "Missing audio dataset!"
    # Check if it has already been processed
    if not os.path.exists(local_paths.DATASET_PATH + "sequence.npy") or \
            not os.path.exists(local_paths.DATASET_PATH + "spectogram.npy") or \
            np.shape(np.load(local_paths.DATASET_PATH + "sequence.npy"))[1] != hparams['max_sentence_length'] or \
            np.shape(np.load(local_paths.DATASET_PATH + "spectogram.npy"))[1:3] != (hparams['max_output_length'], hparams['frequency_bins']):
        # process the data
        tacotron.utils.process_data(local_paths.DATASET_PATH, hparams)
    # Load dataset
    training_sequences, training_spectograms = tacotron.utils.load_dataset(local_paths.DATASET_PATH)

    # limit the dataset such that the model still runs
    training_sequences = training_sequences[0:500,:]
    training_spectograms = training_spectograms[0:500,:]
    improved_tacotron_2_model.train(training_sequences, training_spectograms)
else:
    improved_tacotron_2_model.train_test()
# improved_tacotron_2_model.train_test()

# TODO: save model

improved_tacotron_2_model.predict("This course is fun.")


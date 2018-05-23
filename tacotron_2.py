# Tacotron 2 Main File
import dataset
from tacotron.model import TTS, TTS_Mode
import tacotron
import os
import numpy as np
import local_paths


## TODO:
# - fix postnet
# - check attention
# - save/restore model functionality
# - combine dataset from two random files in train
# - randomize dataset in train


# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data



### Setup the network mode
# Examples (all BASIC components are always enabled/cannot be disabled)
# Just add convolution: mode = TTS_MODE.CONVOLUTIONAL
# Combine Convoluition and 2 Layer LSTM: mode = TTS_MODE.CONVOLUTIONAL | TTS_MODE.TWO_LSTM_DECODER

mode = TTS_Mode.BASIC | TTS_Mode.BIDIRECTIONAL_LSTM_ENCODER | TTS_Mode.TWO_LSTM_DECODER | TTS_Mode.CONVOLUTIONAL

# activate/deactivate test mode, will skip the dataset loading
test = True

databatch_size = 1000

if test:
    # TEST PARAMS
    hparams = {}
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 512
    hparams['max_sentence_length'] = 8
    hparams['basic_lstm_cells'] = 512
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 2
    hparams['frequency_bins'] = 128
    hparams['prenet_cells'] = 128
    hparams['max_output_length'] = 30
    hparams['max_gradient_norm'] = 5
    hparams['learning_rate'] = 1e-4
    hparams['batch_size'] = 64
    hparams['number_conv_layers_encoder'] = 3
    hparams['number_conv_layers_postnet'] = 5
    hparams['is_Training'] = True
    hparams['scale_factor'] = 5000
    hparams['attention_cells'] = 128
else:
    # BATCH PARAMS
    hparams = {}
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 512
    hparams['max_sentence_length'] = 10
    hparams['basic_lstm_cells'] = 1024
    hparams['prenet_cells'] = 128
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 2
    hparams['frequency_bins'] = 128
    hparams['max_output_length'] = 50
    hparams['learning_rate'] = 1e-4
    hparams['batch_size'] = 16
    hparams['number_conv_layers_encoder'] = 3
    hparams['number_conv_layers_postnet'] = 5
    hparams['is_Training'] = True
    hparams['scale_factor'] = 5000
    hparams['attention_cells'] = 128

with open(local_paths.PARAMS_PATH, "w") as file:
    file.write(str(hparams))


improved_tacotron_2_model = TTS(hparams, mode)

if test is False:
    # First check if dataset is available, otherwise cancel
    assert os.path.exists(local_paths.DATASET_PATH + "prompts.data"), "Missing text dataset!"
    assert os.path.exists(local_paths.DATASET_PATH + "wavn"), "Missing audio dataset!"
    # Check if it has already been processedf
    if not os.path.exists(local_paths.DATASET_PATH_PROCESSED + "sequence_0.npy") or \
            not os.path.exists(local_paths.DATASET_PATH_PROCESSED + "spectogram_0.npy") or \
            np.shape(np.load(local_paths.DATASET_PATH_PROCESSED + "sequence_0.npy"))[1] != hparams['max_sentence_length'] or \
            np.shape(np.load(local_paths.DATASET_PATH_PROCESSED + "spectogram_0.npy"))[1:3] != (hparams['max_output_length'], hparams['frequency_bins']):
        # process the data
        tacotron.utils.process_data(local_paths.DATASET_PATH, hparams, databatch_size,local_paths.DATASET_PATH_PROCESSED)
    # Load dataset
    # training_sequences, training_spectograms = tacotron.utils.load_dataset(local_paths.DATASET_PATH, 0)

    # limit the dataset such that the model still runs
    # training_sequences = training_sequences[0:100,:]
    # training_spectograms = training_spectograms[0:100,:]
    improved_tacotron_2_model.train(local_paths.DATASET_PATH_PROCESSED)
else:
    improved_tacotron_2_model.train_test()
# improved_tacotron_2_model.train_test()

# TODO: save model

# improved_tacotron_2_model.predict("", local_paths.PREDICT_PATH)


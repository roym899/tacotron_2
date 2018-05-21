# Tacotron 2 Main File
import dataset
import hparameters
from tacotron.model import TTS, TTS_Mode
import tacotron
import os
import numpy as np
import local_paths


### Setup the network mode
# Examples (all BASIC components are always enabled/cannot be disabled)
# Just add convolution: mode = TTS_MODE.CONVOLUTIONAL
# Combine Convoluition and 2 Layer LSTM: mode = TTS_MODE.CONVOLUTIONAL | TTS_MODE.TWO_LSTM_DECODER
mode = TTS_Mode.ALL ^ TTS_Mode.POSTNET ^ TTS_Mode.PRENET | TTS_Mode.ATTENTION

databatch_size = 1000


with open(local_paths.PARAMS_PATH, "w") as file:
    file.write(str(hparameters))

improved_tacotron_2_model = TTS(hparameters.hparams, mode)

if hparameters.test is False:
    # First check if dataset is available, otherwise cancel
    assert os.path.exists(local_paths.DATASET_PATH + "prompts.data"), "Missing text dataset!"
    assert os.path.exists(local_paths.DATASET_PATH + "wavn"), "Missing audio dataset!"
    # Check if it has already been processedf
    if not os.path.exists(local_paths.DATASET_PATH_PROCESSED + "sequence_0.npy") or \
            not os.path.exists(local_paths.DATASET_PATH_PROCESSED + "spectogram_0.npy") or \
            np.shape(np.load(local_paths.DATASET_PATH_PROCESSED + "sequence_0.npy"))[1] != hparameters.hparams['max_sentence_length'] or \
            np.shape(np.load(local_paths.DATASET_PATH_PROCESSED + "spectogram_0.npy"))[1:3] != (hparameters.hparams['max_output_length'], hparameters.hparams['frequency_bins']):
        # process the data
        tacotron.utils.process_data(local_paths.DATASET_PATH, hparameters.hparams, hparameters.hparams['databatch_size'], local_paths.DATASET_PATH_PROCESSED)
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

improved_tacotron_2_model.predict("", local_paths.PREDICT_PATH)


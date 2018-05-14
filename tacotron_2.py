# Tacotron 2 Main File
import dataset
from tacotron.model import TTS
import tacotron

# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data
DATASET_PATH = "../dataset/"


hparams = {}
hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
hparams['embedding_size'] = 512
hparams['max_sentence_length'] = 100
hparams['basic_encoder_lstm_cells'] = 512
hparams['frequency_bins'] = 2048
hparams['max_output_length'] = 100
hparams['max_gradient_norm'] = 5
hparams['learning_rate'] = 10e-3

# Create dataset
# TODO: First check if dataset is available, otherwise cancel


# TODO: Check if it has already been processed


# TODO: process the data


dataset

improved_tacotron_2_model = TTS(hparams, "basic")

improved_tacotron_2_model.train()

# TODO: save model

improved_tacotron_2_model.predict("This course is fun.")
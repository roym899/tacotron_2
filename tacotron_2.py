# Tacotron 2 Main File
import dataset
from tacotron.model import TTS

# assuming the dataset folder structure is
# DATASET_PATH/wavn/*.wav
# DATASET_PATH/prompts.data
DATASET_PATH = "../dataset/"

# define the basic vocabulary
VOCAB = "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz -?!.,;:\'"

hparams = {}
hparams['src_vocab_size'] = len(VOCAB)
hparams['embedding_size'] = 512
hparams['max_sentence_length'] = 50
hparams['basic_encoder_lstm_cells'] = 256
hparams['frequency_bins'] = 2048
hparams['max_output_length'] = 5000
hparams['max_gradient_norm'] = 5
hparams['learning_rate'] = 0.0001

improved_tacotron_2_model = TTS(hparams, "basic")
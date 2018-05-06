import tensorflow as tf

def load_dataset(path):
    """
    assuming the dataset folder structure is
    # DATASET_PATH/wavn/*.wav
    # DATASET_PATH/prompts.data
    """

    # TODO: Read and parse prompts.data
    # maybe use tf.data.TextLineDataset

    # TODO: learn character embedding of the input sentences
    # Character Embedding in Tensorflow
    # https://www.tensorflow.org/versions/master/programmers_guide/embedding
    # https://www.tensorflow.org/versions/master/tutorials/word2vec

    # TODO: Read the corresponding wav file and link them to the corresponding prompts

    # depending on how long all of this takes, save the resulting dataset and reload it only to skip the computation

    return character_embeddings, mel_spectograms, waveform_samples


def mel_spectogram(wav_data):
    # Maybe use tf.contrib.signal.linear_to_mel_weight_matrix
    return


# TODO: properly setup from the data
# https://www.tensorflow.org/get_started/feature_columns
# https://www.tensorflow.org/get_started/datasets_quickstart
def tacotron2_input_fn(character_embeddings, mel_spectograms, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def wavenet_input_fn(mel_spectograms, waveform_samples, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
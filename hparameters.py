import tacotron.utils

# activate/deactivate test mode, will skip the dataset loading
test = False


if test:
    # TEST PARAMS
    hparams = {}
    hparams['databatch_size'] = 1000
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 512
    hparams['max_sentence_length'] = 80
    hparams['basic_lstm_cells'] = 512
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 8
    hparams['frequency_bins'] = 256
    hparams['prenet_cells'] = 128
    hparams['max_output_length'] = 125
    hparams['max_gradient_norm'] = 5
    hparams['learning_rate'] = 1e-3
    hparams['batch_size'] = 64
    hparams['number_conv_layers_encoder'] = 3
    hparams['number_conv_layers_postnet'] = 5
    hparams['is_Training'] = True
    hparams['scale_factor'] = 1000
    hparams['attention_cells'] = 128

    hparams['smoothing'] = True  # Whether to smooth the attention normalization function
    hparams['attention_dim'] = 128  # dimension of attention space
    hparams['attention_filters'] = 32  # number of attention convolution filters
    hparams['attention_kernel'] = (31,)  # kernel size of attention convolution
    hparams['cumulative_weights'] = True  # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)
    hparams['mask_encoder'] = True  # whether to mask encoder padding while computing attention
else:
    # BATCH PARAMS
    hparams = {}
    hparams['databatch_size'] = 1000
    hparams['src_vocab_size'] = len(tacotron.utils.VOCAB)
    hparams['embedding_size'] = 512
    hparams['max_sentence_length'] = 120
    hparams['basic_lstm_cells'] = 512
    hparams['prenet_cells'] = 128
    hparams['fftsize'] = 2048
    hparams['hops'] = 2048 // 8
    hparams['frequency_bins'] = 256
    hparams['max_output_length'] = 250
    hparams['learning_rate'] = 10e-3
    hparams['batch_size'] = 16
    hparams['number_conv_layers_encoder'] = 3
    hparams['number_conv_layers_postnet'] = 5
    hparams['is_Training'] = True
    hparams['scale_factor'] = 1000
    hparams['attention_cells'] = 128

    hparams['smoothing'] = True  # Whether to smooth the attention normalization function
    hparams['attention_dim'] = 128  # dimension of attention space
    hparams['attention_filters'] = 32  # number of attention convolution filters
    hparams['attention_kernel'] = (31,)  # kernel size of attention convolution
    hparams['cumulative_weights'] = True  # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)
    hparams['mask_encoder'] = True  # whether to mask encoder padding while computing attention

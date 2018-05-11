import tensorflow as tf
import tacotron.utils
import numpy as np

class TTS(object):
    def __init__(self, hparams, mode):
        if mode == "basic":
            # embedding -> LSTM layer -> LSTM layer -> Dense
            # -----------------------    -------------------
            #         Encoder               Decoder

            self.hparams = hparams

            # define the input
            self.encoder_inputs = tf.placeholder(tf.int32, [hparams['max_sentence_length'], 1], 'inputs')

            # Embedding
            embedding_encoder = tf.get_variable(
                "embedding_encoder", [hparams['src_vocab_size'], hparams['embedding_size']])
            # Look up embedding:
            #   encoder_inputs: [max_time, batch_size]
            #   encoder_emb_inp: [max_time, batch_size, embedding_size]
            self.encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)

            # Build RNN cell
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_encoder_lstm_cells'])
            # Run Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            self.encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                       self.encoder_emb_inp,
                                                                       sequence_length=[hparams['max_sentence_length']],
                                                                       dtype=tf.float32,
                                                                       time_major=True)

            # Build RNN cell
            # Helper
            # decoder_outputs = tf.placeholder(tf.float64,
            #                                 [hparams['max_output_length'], hparams['frequency_bins']],
            #                                 'inputs')
            #
            # helper = tf.contrib.seq2seq.TrainingHelper(decoder_outputs, [hparams['max_output_length']], time_major=True)
            # # Decoder
            # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_encoder_lstm_cells'])
            # projection_layer = tf.layers.Dense(hparams['frequency_bins'], use_bias=False)
            # decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
            #                                           helper,
            #                                           encoder_state,
            #                                           output_layer=projection_layer)
            # # Dynamic decoding
            # self.outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            # spectograms = self.outputs.rnn_output
            #
            # target_spectograms = dataset_iterator.target_spectograms
            # train_loss = tf.losses.mean_squared_error(target_spectograms, spectograms)
            # params = tf.trainable_variables()
            # gradients = tf.gradients(train_loss, params)
            # clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams['max_gradient_norm'])

            # Optimization
            # optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])
            # update_step = optimizer.apply_gradients((clipped_gradients, params))

        elif mode == "tacotron-2":
            pass

    def train(self):
        pass

    def predict(self, text):
        init = tf.global_variables_initializer()
        input = tacotron.utils.text_to_sequence(text, self.hparams['max_sentence_length'])
        training_session = tf.Session()
        training_session.run(init)
        res = training_session.run(self.encoder_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input,1)})
        test_out = res[:,0,:]
        print("test")

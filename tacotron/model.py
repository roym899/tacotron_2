import tensorflow as tf
import tacotron.utils
import numpy as np
from wavenet import wavenet
from tacotron.RegressionHelper import RegressionHelper
import matplotlib.pyplot as plt
import matplotlib

class TTS(object):
    def __init__(self, hparams, mode):
        def sample_fcn(outputs):
            return outputs
        if mode == "basic":


            # embedding -> LSTM layer -> LSTM layer -> Dense
            # -----------------------    -------------------
            #         Encoder               Decoder

            self.hparams = hparams

            # define the input
            self.encoder_inputs = tf.placeholder(tf.int32, [None, hparams['max_sentence_length']], 'inputs')
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')

            batch_size = tf.shape(self.encoder_inputs)[0]

            # Embedding
            embedding_encoder = tf.get_variable(
                "embedding_encoder", [hparams['src_vocab_size'], hparams['embedding_size']])
            # Look up embedding:
            #   encoder_inputs: [batch_size, max_time]
            #   encoder_emb_inp: [batch_size, max_time, embedding_size]
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
                                                                       time_major=False)

            # Build RNN cell
            # Helper
            self.target_spectograms = tf.placeholder(tf.float32, [None, hparams['max_output_length'], hparams['frequency_bins']])

            # Decoder
            # self.global_step_tensor = tf.Variable(10, trainable=False, name='global_step')

            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_encoder_lstm_cells'])
            projection_layer = tf.layers.Dense(hparams['frequency_bins'], use_bias=False)
            training_helper = tf.contrib.seq2seq.TrainingHelper(self.target_spectograms,
                                                      [hparams['max_output_length']], # TODO: adjust to batch size
                                                      time_major=False)
            inference_helper = RegressionHelper(batch_size, self.hparams['frequency_bins'], self.hparams['max_output_length'])

            # helper = tf.cond(self.is_training,
            #                  lambda: tf.contrib.seq2seq.TrainingHelper(self.target_spectograms,
            #                                                            [hparams['max_output_length']],
            #                                                            time_major=False),
            #                  lambda: RegressionHelper(batch_size, hparams['frequency_bins'],
            #                                           hparams['max_output_length']))


            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      training_helper,
                                                      encoder_state,
                                                      output_layer=projection_layer)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                inference_helper,
                                                                encoder_state,
                                                                output_layer=projection_layer)

            # Dynamic decoding
            self.inference_outputs = tf.contrib.seq2seq.dynamic_decode(inference_decoder)
            self.training_outputs = tf.contrib.seq2seq.dynamic_decode(training_decoder)

            # Optimization
            inference_spectograms = self.inference_outputs[0].rnn_output
            training_spectograms = self.training_outputs[0].rnn_output
            # target_spectograms = dataset_iterator.target_spectograms
            self.train_loss = tf.losses.mean_squared_error(self.target_spectograms, training_spectograms)

            optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])
            # clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams['max_gradient_norm'])
            self.minimize = optimizer.minimize(self.train_loss)

            self.session = tf.Session()
            init = tf.global_variables_initializer()
            self.session.run(init)



        elif mode == "tacotron-2":
            pass

    def train(self):
        training_sentence = "Your father may complain about how badly lit the local roads are."
        input = tacotron.utils.text_to_sequence(training_sentence, self.hparams['max_sentence_length'])
        pre_res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input,0), self.is_training:False})

        plt.imshow(pre_res[0].rnn_output[0,:,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
        plt.show()

        fftsize = 2048
        hops = fftsize//8
        audio = wavenet.load_audio("/home/leo/dd2424/project/dataset/wavn/APDC2-070-02.wav", expected_samplerate=16000)
        training_spectogram = wavenet.calculate_stft(audio, fftsize, hops)
        training_spectogram = abs(training_spectogram) ** 2
        # training_spectogram = np.log(training_spectogram)
        training_spectogram = training_spectogram[0:self.hparams['max_output_length'],:]

        rec_audio = wavenet.reconstruct_signal(training_spectogram, fftsize, hops, 100)
        max_value = np.max(abs(rec_audio))
        if max_value > 1.0:
            rec_audio = rec_audio / max_value
        audio = wavenet.save_audio(rec_audio, 16000, "/home/leo/dd2424/project/dataset/test_training.wav")


        # training_spectogram[training_spectogram<5000]=0
        # training_spectogram[training_spectogram>=5000]=5000
        # print(np.mean(np.power(training_spectogram, 2)))
        # print(np.count_nonzero(np.power(training_spectogram, 2)))
        # print(np.sum(np.power(training_spectogram, 2)))



        # training_spectogram = np.pad(training_spectogram, ((0,self.hparams['max_output_length']-training_spectogram.shape[0]), (0,0)), 'constant')
        # training_spectogram = (training_spectogram - np.min(training_spectogram) ) / (np.max(training_spectogram)-np.min(training_spectogram))
        training_spectogram = np.expand_dims(training_spectogram, 0)
        plt.imshow(training_spectogram[0, :, :], cmap='hot', interpolation='nearest',norm=matplotlib.colors.Normalize())
        plt.show()
        training_sequence = tacotron.utils.text_to_sequence(training_sentence, self.hparams['max_sentence_length'])
        loss = np.Inf
        counter = 0
        next_image_loss = 0
        test = 0
        while loss>0:
            loss, opt = self.session.run([self.train_loss, self.minimize], feed_dict={self.encoder_inputs: np.expand_dims(training_sequence,0), self.target_spectograms: training_spectogram, self.is_training:True})
            print("Loss: {}".format(loss))
            counter += 1
            if loss < next_image_loss or counter > 500:
                post_res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input, 0), self.is_training: False})

                plt.imshow(post_res[0].rnn_output[0,:,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
                plt.show()
                # next_image_loss = loss - 1
                counter = 0

                rec_audio = wavenet.reconstruct_signal(post_res[0].rnn_output[0,:,:], fftsize, hops, 100)
                max_value = np.max(abs(rec_audio))
                if max_value > 1.0:
                    rec_audio = rec_audio / max_value
                audio = wavenet.save_audio(rec_audio, 16000, "/home/leo/dd2424/project/dataset/test_{}.wav".format(test))
                test += 1


    def predict(self, text):
        input = tacotron.utils.text_to_sequence(text, self.hparams['max_sentence_length'])
        res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input,0), self.is_training:False})
        # test_out = res[0,:,:]
        # print('global_step: %s' % tf.train.global_step(self.session, self.global_step_tensor))

        # TODO: convert output to audio using griffith lim
        print(res[0].rnn_output[0,0,180])
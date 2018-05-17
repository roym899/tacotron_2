import tensorflow as tf
import tacotron.utils
import numpy as np
from wavenet import wavenet
from tacotron.RegressionHelper import RegressionHelper
from tacotron.PrenetTrainingHelper import PrenetTrainingHelper
import matplotlib.pyplot as plt
import matplotlib
import local_paths
from enum import IntFlag

import random

class TTS_Mode(IntFlag):
     BASIC = 0 # Embedding, Onedirectional LSTM, Onedirectional LSTM, Projection
     CONVOLUTIONAL = 1
     TWO_LSTM_DECODER = 2
     PRENET = 4
     POSTNET = 8
     BIDIRECTIONAL_LSTM_ENCODER = 16
     ATTENTION = 32
     ALL = 63

class TTS(object):


    def __init__(self, hparams, mode):
        def sample_fcn(outputs):
            return outputs

        # embedding -> 3 Convolutional Layers -> LSTM layer -> LSTM layer -> Dense
        # -------------------------------------------------    -------------------
        #                        Encoder                            Decoder

        self.hparams = hparams

        # define the input
        with tf.name_scope('inputs'):
            self.encoder_inputs = tf.placeholder(tf.int32, [None, hparams['max_sentence_length']], 'inputs')
        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        batch_size = tf.shape(self.encoder_inputs)[0]

        # Embedding
        embedding_encoder = tf.get_variable(
            "embedding_encoder", [hparams['src_vocab_size'], hparams['embedding_size']])
        self.encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs) # [batch_size, max_time, embedding_size]

        # Convolutional Layers
        if mode & TTS_Mode.CONVOLUTIONAL:
            encoder_input = self.conv_encoder(self.encoder_emb_inp,hparams['number_conv_layers_encoder'],hparams['is_Training'])
        else:
            encoder_input = self.encoder_emb_inp

        if mode & TTS_Mode.BIDIRECTIONAL_LSTM_ENCODER:
            encoder_cell_forward = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_lstm_cells']//2)
            encoder_cell_backward = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_lstm_cells']//2)
            self.encoder_outputs, encoder_state_output, encoder_state_output_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                [encoder_cell_forward],
                [encoder_cell_backward],
                encoder_input,
                sequence_length=tf.fill([batch_size], hparams['max_sentence_length']),
                dtype=tf.float32,
                time_major=False)
        else:
            # Build RNN cell
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_lstm_cells'])
            # Run Dynamic RNN
            self.encoder_outputs, encoder_state_output = tf.nn.dynamic_rnn(encoder_cell,
                                                                           encoder_input,
                                                                           sequence_length=tf.fill([batch_size],hparams['max_sentence_length']),
                                                                           dtype=tf.float32,
                                                                           time_major=False)
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]

        # Build RNN cell
        # Helper
        self.target_spectograms = tf.placeholder(tf.float32,
                                                 [None, hparams['max_output_length'], hparams['frequency_bins']])


        # Attention
        if mode & TTS_Mode.ATTENTION:
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention()
            pass
        else:
            pass


        # Decoder
        # self.global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
        projection_layer = tf.layers.Dense(hparams['frequency_bins'], use_bias=False, activation=tf.nn.relu)
        if mode & TTS_Mode.TWO_LSTM_DECODER:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hparams['basic_lstm_cells']//2) for i in range(2)]
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            if mode & TTS_Mode.BIDIRECTIONAL_LSTM_ENCODER:
                decoder_initial_state = list()
                c1 = encoder_state_output[0][0]
                c2 = encoder_state_output_back[0][0]

                h1 = encoder_state_output[0][1]
                h2 = encoder_state_output_back[0][1]
                decoder_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c1, h1))
                decoder_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c2, h2))
                decoder_initial_state = tuple(decoder_initial_state)
            else:
                decoder_initial_state = list()
                c = encoder_state_output[0]
                c1 = c[:,0:hparams['basic_lstm_cells']]
                c2 = c[:,hparams['basic_lstm_cells']:]
                h = encoder_state_output[1]
                h1 = h[:,0:hparams['basic_lstm_cells']]
                h2 = h[:,hparams['basic_lstm_cells']:]
                decoder_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c1, h1))
                decoder_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c2, h2))
                decoder_initial_state = tuple(decoder_initial_state)

        else:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams['basic_lstm_cells'])
            if mode & TTS_Mode.BIDIRECTIONAL_LSTM_ENCODER:
                decoder_initial_state = list()
                c1 = encoder_state_output[0][0]
                c2 = encoder_state_output_back[0][0]
                c = tf.concat([c1,c2],1)

                h1 = encoder_state_output[0][1]
                h2 = encoder_state_output_back[0][1]
                h = tf.concat([h1,h2],1)

                decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c,h)
            else:
                decoder_initial_state = encoder_state_output


        if mode & TTS_Mode.PRENET:
            pre = []
            pre.append(tf.layers.Dense(hparams['prenet_cells'], use_bias=False, activation=tf.nn.relu))
            pre.append(tf.layers.Dense(hparams['prenet_cells'], use_bias=False, activation=tf.nn.relu))

            inference_helper = RegressionHelper(batch_size,
                                                self.hparams['frequency_bins'],
                                                hparams['max_output_length'],
                                                pre)
            training_helper = PrenetTrainingHelper(self.target_spectograms,
                                                   tf.fill([batch_size], hparams['max_output_length']),
                                                   pre,
                                                   time_major=False)
        else:
            training_helper = tf.contrib.seq2seq.TrainingHelper(self.target_spectograms,
                                                                tf.fill([batch_size], hparams['max_output_length']),
                                                                time_major=False)
            inference_helper = RegressionHelper(batch_size,
                                                self.hparams['frequency_bins'],
                                                hparams['max_output_length'])



        # helper = tf.cond(self.is_training,
        #                  lambda: tf.contrib.seq2seq.TrainingHelper(self.target_spectograms,
        #                                                            [hparams['max_output_length']],
        #                                                            time_major=False),
        #                  lambda: RegressionHelper(batch_size, hparams['frequency_bins'],
        #                                           hparams['max_output_length']))

        training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                           training_helper,
                                                           decoder_initial_state,
                                                           output_layer=projection_layer)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                            inference_helper,
                                                            decoder_initial_state,
                                                            output_layer=projection_layer)

        # Dynamic decoding
        self.inference_outputs = tf.contrib.seq2seq.dynamic_decode(inference_decoder)
        self.training_outputs = tf.contrib.seq2seq.dynamic_decode(training_decoder)

        # Optimization
        inference_spectograms = self.inference_outputs[0].rnn_output
        training_spectograms = self.training_outputs[0].rnn_output
        # target_spectograms = dataset_iterator.target_spectograms
        self.train_loss = tf.losses.mean_squared_error(self.target_spectograms, training_spectograms)
        tf.summary.scalar('train loss', self.train_loss)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = hparams['learning_rate']
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100, 0.96, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.train_loss, global_step=global_step)

        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)


    def conv_encoder(self, conv_input,number_conv_layers,is_training, kernel_size = (5,), channels = 512,activation = tf.nn.relu ):
        """
        Calculates the output of the 1D convolutional layers for the encoder part of the network.
        :param conv_input: Input to the first convolutional layer
        :param number_conv_layers: Defiens of how many convolutional layers this part of the network consists
        :param is_training: Parameter that indicates if the network is in training or inference mode.
                            Determines how the batch normalization after every convolutional layer is performed
        :param kernel_size: Defines the size of the kernel that is applied at every layer
        :param channels:    Defines the dimension of the character embedding and helps to perform the 1D convolution
        :param activation:  Defines the activation function of every convolutional layer
        :return: Returns the output after number_conv_layers many convolutional layers.
                 Before returning the shape of the output is adjusted to fit the expected shape of the encoder
        """
        conv_output = conv_input
        for i in range(number_conv_layers):
            conv_output = tf.layers.conv1d(conv_output,filters=channels,kernel_size=kernel_size,activation=None,padding='same')
            batched_output = tf.layers.batch_normalization(conv_output,training=is_training)
            activated_output = activation(batched_output)
            conv_output = activated_output
        conv_shape = conv_output.get_shape().as_list()
        conv_output = tf.reshape(conv_output,[-1,conv_shape[1],conv_shape[2]])
        return conv_output


    def train_test(self):
        training_sentence = "Your father may complain how badly lid the local roads are."
        predict_sentence = "Your father may."
        input = tacotron.utils.text_to_sequence(training_sentence, self.hparams['max_sentence_length'])
        predict_input = tacotron.utils.text_to_sequence(predict_sentence, self.hparams['max_sentence_length'])
        pre_res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input,0), self.is_training:False})

        plt.imshow(pre_res[0].rnn_output[0,:,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
        plt.show()

        fftsize = 2048
        hops = 512
        audio = wavenet.load_audio(local_paths.TEST_AUDIO, expected_samplerate=16000)
        training_spectogram = wavenet.calculate_stft(audio, fftsize, hops)
        training_spectogram = abs(training_spectogram) ** 2
        # training_spectogram = training_spectogram[0:self.hparams['max_output_length'],:]
        training_spectogram[:, self.hparams['frequency_bins']:-1] = 0
        training_spectogram_test =  np.copy(training_spectogram)
        rec_audio = wavenet.reconstruct_signal(training_spectogram_test, fftsize, hops, 100)
        max_value = np.max(abs(rec_audio))
        if max_value > 1.0:
            rec_audio = rec_audio / max_value
        audio = wavenet.save_audio(rec_audio, 16000, local_paths.RECONSTRUCTED_AUDIO_OUTPUT)

        training_spectogram = training_spectogram[0:self.hparams['max_output_length'], 0:self.hparams['frequency_bins']]/1000
        # training_spectogram = np.log(training_spectogram)
        # training_spectogram = np.pad(training_spectogram, ((0,self.hparams['max_output_length']-training_spectogram.shape[0]), (0,0)), 'constant')
        # training_spectogram = (training_spectogram - np.min(training_spectogram) ) / (np.max(training_spectogram)-np.min(training_spectogram))
        training_spectogram = np.expand_dims(training_spectogram, 0)
        print("Min: {}, Max: {}".format(np.min(training_spectogram), np.max(training_spectogram)))
        plt.imshow(training_spectogram[0, :, :], cmap='hot', interpolation='nearest',norm=matplotlib.colors.Normalize())
        plt.show()
        training_sequence = tacotron.utils.text_to_sequence(training_sentence, self.hparams['max_sentence_length'])
        loss = np.Inf
        counter = 0
        next_image_loss = 0
        test = 0
        epochs = 0
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("/tmp",sess.graph)

        while loss>0:
            summary, loss, opt = self.session.run([merged, self.train_loss, self.optimizer], feed_dict={self.encoder_inputs: np.expand_dims(training_sequence,0), self.target_spectograms: training_spectogram, self.is_training:True})
            writer.add_summary(summary, epochs)
            epochs += 1
            # if epochs == 10:
            #     break
            print("Loss: {}".format(loss))
            counter += 1
            if loss < next_image_loss or counter > 100:
                post_res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(predict_input, 0), self.is_training: False})

                print("Min: {}, Max: {}".format(np.min(post_res[0].rnn_output[0,:,:]), np.max(post_res[0].rnn_output[0,:,:])))
                plt.imshow(post_res[0].rnn_output[0,:,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
                plt.show()
                # next_image_loss = loss - 1
                counter = 0

                training_spectogram_test[:, 0:self.hparams['frequency_bins']] = post_res[0].rnn_output[0,:,:]*1000
                rec_audio = wavenet.reconstruct_signal(training_spectogram_test, fftsize, hops, 100)
                max_value = np.max(abs(rec_audio))
                if max_value > 1.0:
                    rec_audio = rec_audio / max_value

                audio = wavenet.save_audio(rec_audio, 16000, local_paths.TEST_PATTERN.format(test))
                test += 1


    def train(self, training_sequences, training_spectograms):
        loss = np.Inf
        counter = 0
        next_image_loss = 0
        test = 0
        test_sentence = training_sequences[0,:]
        test_spectogram = training_spectograms[0,:,:]
        plt.imshow(test_spectogram, cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
        plt.show()
        batch_size = self.hparams['batch_size']
        counter = 0
        while loss>0:
            data = (training_sequences, training_spectograms)
            idx = 0
            while idx+batch_size < data[0].shape[0]:
                loss, opt = self.session.run([self.train_loss, self.optimizer],
                                             feed_dict={self.encoder_inputs: data[0][idx:idx+batch_size,:],
                                                        self.target_spectograms: data[1][idx:idx+batch_size,:,:],
                                                        self.is_training:True})
                idx += batch_size
            loss = self.session.run(self.train_loss,
                                    feed_dict={self.encoder_inputs: training_sequences,
                                               self.target_spectograms: training_spectograms,
                                               self.is_training: True})

            print("Loss: {}".format(loss))
            counter += 1
            if loss < next_image_loss or counter > 50:
                post_res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(test_sentence, 0), self.is_training: False})

                print("Min: {}, Max: {}".format(np.min(post_res[0].rnn_output[0,:,:]), np.max(post_res[0].rnn_output[0,:,:])))
                plt.imshow(post_res[0].rnn_output[0,:,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize())
                plt.show()
                # next_image_loss = loss - 1
                counter = 0



    def predict(self, text):
        input = tacotron.utils.text_to_sequence(text, self.hparams['max_sentence_length'])
        res = self.session.run(self.inference_outputs, feed_dict={self.encoder_inputs: np.expand_dims(input,0), self.is_training:False})
        # test_out = res[0,:,:]
        # print('global_step: %s' % tf.train.global_step(self.session, self.global_step_tensor))

        # TODO: convert output to audio using griffith lim
        #print(res[0].rnn_output[0,0,180])
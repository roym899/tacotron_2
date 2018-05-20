import tensorflow as tf

class Prenet:
    """Two fully connected layers used as an information bottleneck for the attention.
    """

    def __init__(self, is_training, layer_sizes=[256, 256], activation=tf.nn.relu, scope=None):
        """
        Args:
            is_training: Boolean, determines if the model is in training or inference to control dropout
            layer_sizes: list of integers, the length of the list represents the number of pre-net
                layers and the list values represent the layers number of units
            activation: callable, activation functions of the prenet layers.
            scope: Prenet scope.
        """
        super(Prenet, self).__init__()
        # self.drop_rate = hparams.tacotron_dropout_rate

        self.layer_sizes = layer_sizes
        self.is_training = is_training
        self.activation = activation

        self.scope = 'prenet' if scope is None else scope

    def __call__(self, inputs):
        x = inputs

        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layer_sizes):
                dense = tf.layers.dense(x, units=size, activation=self.activation,
                                        name='dense_{}'.format(i + 1))
                # The paper discussed introducing diversity in generation at inference time
                # by using a dropout of 0.5 only in prenet layers (in both training and inference).
                # x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
                #                       name='dropout_{}'.format(i + 1) + self.scope)
                x = dense
        return x
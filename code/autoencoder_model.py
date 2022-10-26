import tensorflow as tf
import tensorflow.keras.backend as K

class Autoencoder(tf.keras.Model):
  def __init__(self, input_dims, embedding_size):
    super(Autoencoder, self).__init__()
    self.bce = tf.keras.losses.BinaryCrossentropy()
    self.loss = tf.keras.losses.BinaryCrossentropy() #self.repeat_loss
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002)
    self.batch_size = 128
    self.rnn_size = embedding_size
    self.input_dims = input_dims

    self.encoder = tf.keras.Sequential(layers=[
      tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(self.rnn_size)),
    ])
    self.decoder = tf.keras.Sequential(layers=[
      tf.keras.layers.RepeatVector(input_dims[0]),
      tf.keras.layers.LSTM(self.rnn_size, return_sequences=True),
      tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(input_dims[1])
      ),
    ])

  def call(self, x):
    """
    Perform the forward pass
    :param x: the inputs to the model
    :return:  the batch element probabilities as a tensor
    """
    latent = self.encoder(x)
    reconstruction = self.decoder(latent)
    return reconstruction

  def repeat_loss(self, yTrue, yPred):
    alpha = 0.2
    yTrue = K.cast(yTrue, dtype=tf.float32)
    return self.bce(yTrue, yPred)

class AEClassifier(tf.keras.Model):
  def __init__(self, encoder, n_classes):
    super(AEClassifier, self).__init__()
    self.loss = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    self.batch_size = 128
    self.categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    self.auroc = tf.keras.metrics.AUC()
    self.recall = tf.keras.metrics.Recall()
    self.precision = tf.keras.metrics.Precision()
    self.encoder = encoder
    self.h1 = 128
    self.h2 = 64
    self.h3 = 32
    self.num_classes = n_classes

    self.layers1 = tf.keras.Sequential(layers=[
      tf.keras.layers.Dense(self.h1, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.Dense(self.h2, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.Dense(self.h3, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer='random_normal')
    ])

  def call(self, x):
    """
    Perform the forward pass
    :param x: the inputs to the model
    :return:  the batch element probabilities as a tensor
    """
    latent = self.encoder(x)
    probs = self.layers1(latent)
    return probs

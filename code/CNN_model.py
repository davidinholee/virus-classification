import tensorflow as tf

class CNNModel(tf.keras.Model):
  def __init__(self, num_classes):
    # Maybe add batch normalization
    super(CNNModel, self).__init__()
    self.loss = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.003)
    self.categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    self.auroc = tf.keras.metrics.AUC()
    self.recall = tf.keras.metrics.Recall()
    self.precision = tf.keras.metrics.Precision()
    self.batch_size = 128
    self.num_classes = num_classes
    self.h1 = 64
    self.h2 = 32

    self.layers1 = tf.keras.Sequential(layers=[
      tf.keras.layers.Conv1D(filters=32, kernel_size=16, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv1D(filters=8, kernel_size=4, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
      tf.keras.layers.Flatten()
    ])
    self.layers2 = tf.keras.Sequential(layers=[
      tf.keras.layers.Dense(self.h1, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.Dense(self.h2, activation='relu', kernel_initializer='random_normal'),
      tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer='random_normal')
    ])

  def call(self, x, training=True):
    """
    Perform the forward pass
    :param x: the inputs to the model
    :return:  the batch element probabilities as a tensor
    """
    x = self.layers1(x, training=training)
    probs = self.layers2(x)
    return probs

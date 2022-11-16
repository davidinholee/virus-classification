import numpy as np
import tensorflow as tf
from preprocess import get_full_data
from fabijanska_cnn import getNetwork

import matplotlib.pyplot as plt
           
def train(model, dataset, val_dataset, epoch):
  """
  Runs through one epoch of training
  :param model:
  :param dataset:
  :param epoch:
  :return: Epoch loss
  """

  dataset = dataset.shuffle(16000)
  dataset = dataset.batch(model.batch_size)
  tot_loss = 0
  tot_batches = 0
  train_losses = []
  val_losses = []
  # Run through a batch
  for i, batch in enumerate(dataset):
    batch_inputs = batch["data"]
    batch_labels = tf.one_hot(batch["class_id"], depth=model.num_classes)
    with tf.GradientTape() as tape:
      probs = model.call(tf.cast(batch_inputs, tf.float32))
      loss = model.loss(batch_labels, probs)
    tot_loss += loss
    tot_batches += 1
    if i % 50 == 0:
      train_losses.append(loss)
      val_losses.append(validate(model, val_dataset))
      print("Epoch " + str(epoch) + " batch " + str(i) + ", loss=" + str(loss.numpy()) + ", val_loss=" + str(val_losses[-1].numpy()), flush=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return tot_loss/tot_batches, train_losses, val_losses

def validate(model, dataset):
  dataset = dataset.batch(model.batch_size)
  tot_loss = 0
  tot_batches = 0
  for batch in dataset:
    batch_inputs = batch["data"]
    batch_labels = tf.one_hot(batch["class_id"], depth=model.num_classes)
    probs = model.call(tf.cast(batch_inputs, tf.float32))
    loss = model.loss(batch_labels, probs)
    tot_loss += loss
    tot_batches += 1
  return tot_loss/tot_batches

def test(model, dataset):
  """
  Runs through testing set
  :param model:
  :param dataset:
  :return: Confusion matrix of model across all classes
  """
  
  confusion_matrix = np.zeros([model.num_classes, model.num_classes])
  dataset = dataset.batch(model.batch_size)
  # Reset metric states
  model.auroc.reset_state()
  model.recall.reset_state()
  model.precision.reset_state()
  model.categorical_accuracy.reset_state()
  print()
  # Calculate metrics
  for batch in dataset:
    batch_inputs = batch["data"]
    batch_labels = tf.one_hot(batch["class_id"], depth=model.num_classes)
    probs = model.call(tf.cast(batch_inputs, tf.float32), training=False)
    model.auroc.update_state(batch_labels, probs)
    model.recall.update_state(batch_labels, probs)
    model.precision.update_state(batch_labels, probs)
    model.categorical_accuracy.update_state(batch_labels, probs)
    confusion_matrix = confusion_matrix + tf.math.confusion_matrix(batch_labels.numpy().argmax(axis = 1), probs.numpy().argmax(axis = 1), model.num_classes)

  return confusion_matrix

def main():
  num_classes = 4
  epochs = 15
  run_preproc = True
  data_path = "//wsl$/Ubuntu/home/moasi/Singh_Lab/virus-classification/virus-classification/viral data/"
  sequencefiles = [data_path + "covid.fasta", 
    data_path + "influenza.fasta",
    data_path + "mers.fasta",
    data_path + "sars.fasta"]
  class_names = ["covid", "influenza", "mers", "sars"]
    
  #===========================================================================

  # Load data
  if run_preproc:
    train_dataset, val_dataset, test_dataset = get_full_data(sequencefiles, class_names, data_path)
  else:
    type_spec = {"data": tf.TensorSpec(shape=[None, 5], dtype=tf.bool), 
      "class_name": tf.TensorSpec(shape=None, dtype=tf.string), 
      "class_id": tf.TensorSpec(shape=None, dtype=tf.int32)}
    train_dataset = tf.data.experimental.load(data_path + "_train", type_spec)
    val_dataset = tf.data.experimental.load(data_path + "_val", type_spec)
    test_dataset = tf.data.experimental.load(data_path + "_test", type_spec)

  # TODO: Get max length of sequence
  maxLen = 14000
  # Intialize Model
  model = getNetwork(maxLen, num_classes)
  
  # Train model
  train_losses = []
  val_losses = []
  x_ticks = []
  for e in range(epochs):
    loss, t_losses, v_losses = train(model, train_dataset, val_dataset, e)
    train_losses = train_losses + t_losses
    val_losses = val_losses + v_losses
    if e % 3 == 0:
      x_ticks.append("Epoch " + str(e))
    print("Epoch " + str(e) + ", avg loss=" + str(loss.numpy()) + "\n", flush=True)
  xs = range(len(train_losses))
  plt.plot(xs, train_losses, label="train loss")
  plt.plot(xs, val_losses, label="val loss")
  plt.xticks(range(0, len(train_losses), int(len(train_losses)/(epochs/3))), x_ticks)
  plt.ylabel("loss")
  plt.title("Train/Val Loss per Epoch")
  plt.legend()
  plt.savefig("./results/trainval_loss_cnn.png")

  # Print model metrics on train set
  confusion_matrix = test(model, train_dataset)
  precision = model.precision.result().numpy()
  recall = model.recall.result().numpy()
  print("Train Accuracy: " + str(model.categorical_accuracy.result().numpy()))
  print("Train AUROC: " + str(model.auroc.result().numpy()))
  print("Train Precision: " + str(precision))
  print("Train Recall: " + str(recall))
  print("Train F1 score: " + str(2 * recall * precision / (precision + recall)))
  print("Train Confusion Matrix: " + str(confusion_matrix))

  # Print model metrics on test set
  confusion_matrix = test(model, test_dataset)
  precision = model.precision.result().numpy()
  recall = model.recall.result().numpy()
  print("Test Accuracy: " + str(model.categorical_accuracy.result().numpy()))
  print("Test AUROC: " + str(model.auroc.result().numpy()))
  print("Test Precision: " + str(precision))
  print("Test Recall: " + str(recall))
  print("Test F1 score: " + str(2 * recall * precision / (precision + recall)))
  print("Test Confusion Matrix: " + str(confusion_matrix))

  # Print model summary
  model.build((None, 14000, 5))
  print()
  print(model.summary())


if __name__ == "__main__":
  main()

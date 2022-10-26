import numpy as np
import tensorflow as tf
from preprocess_final import get_full_data
from CNN_model import CNNModel
from autoencoder_model import Autoencoder, AEClassifier
import matplotlib.pyplot as plt
           
def train_ae(model, dataset, val_dataset, epoch):
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
    batch_labels = batch_inputs #tf.one_hot(batch["class_id"], depth=model.num_classes)
    with tf.GradientTape() as tape:
      probs = model.call(tf.cast(batch_inputs, tf.float32))
      loss = model.loss(batch_labels, probs)
    tot_loss += loss
    tot_batches += 1
    if i % 50 == 0:
      train_losses.append(loss)
      val_losses.append(validate(model, val_dataset, ae=True))
      print("Epoch " + str(epoch) + " batch " + str(i) + ", loss=" + str(loss.numpy()) + ", val_loss=" + str(val_losses[-1].numpy()), flush=True)
    if i % 1000 == 0:
      input = "".join([str(nt.numpy()) for nt in tf.argmax(batch_inputs[0], axis=1)]).replace("0", "A").replace("1", "C").replace("2", "G").replace("3", "T").replace("4", "N")
      recon = "".join([str(nt.numpy()) for nt in tf.argmax(probs[0], axis=1)]).replace("0", "A").replace("1", "C").replace("2", "G").replace("3", "T").replace("4", "N")
      print("Input:", input)
      print("Recon:", recon)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return tot_loss/tot_batches, train_losses, val_losses

def train_classifier(model, dataset, val_dataset, epoch):
  """
  Runs through one epoch of training
  :param model:
  :param dataset:
  :param epoch:
  :return: Epoch loss
  """

  dataset = dataset.shuffle(8192)
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
      val_losses.append(validate(model, val_dataset, ae=False))
      print("Epoch " + str(epoch) + " batch " + str(i) + ", loss=" + str(loss.numpy()) + ", val_loss=" + str(val_losses[-1].numpy()), flush=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return tot_loss/tot_batches, train_losses, val_losses

def validate(model, dataset, ae):
  dataset = dataset.batch(model.batch_size)
  tot_batches = 0
  tot_loss = 0
  for batch in dataset:
    batch_inputs = batch["data"]
    if ae:
      batch_labels = tf.cast(batch_inputs, tf.float32)
    else:
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
  # Reset model state
  model.auroc.reset_state()
  model.recall.reset_state()
  model.precision.reset_state()
  model.categorical_accuracy.reset_state()
  print()
  # Calculate metrics
  for batch in dataset:
    batch_inputs = batch["data"]
    batch_labels = tf.one_hot(batch["class_id"], depth=model.num_classes)
    probs = model.call(tf.cast(batch_inputs, tf.float32))
    model.auroc.update_state(batch_labels, probs)
    model.recall.update_state(batch_labels, probs)
    model.precision.update_state(batch_labels, probs)
    model.categorical_accuracy.update_state(batch_labels, probs)
    confusion_matrix = confusion_matrix + tf.math.confusion_matrix(batch_labels.numpy().argmax(axis = 1), probs.numpy().argmax(axis = 1), model.num_classes)

  return confusion_matrix

def main():
  num_classes = 4
  epochs_ae = 8
  epochs_dense = 12
  run_autoencoder = True
  run_preproc = False
  data_path = "./fasta_data_final/dataset"
  sequencefiles = ["./fasta_data_final/covid.fasta", 
    "./fasta_data_final/influenza.fasta",
    "./fasta_data_final/mers.fasta", 
    "./fasta_data_final/sars.fasta"]
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
            
  # Intialize Autoencoder
  if run_autoencoder:
    for x in train_dataset:
      shape = x["data"].shape[-2:]
      break
    ae_model = Autoencoder(shape, embedding_size=16)
    
    # Train model
    train_losses = []
    val_losses = []
    x_ticks = []
    for e in range(epochs_ae):
      loss, t_losses, v_losses = train_ae(ae_model, train_dataset, val_dataset, e)
      train_losses = train_losses + t_losses
      val_losses = val_losses + v_losses
      x_ticks.append("Epoch " + str(e))
      print("Epoch " + str(e) + ", avg loss=" + str(loss.numpy()) + "\n", flush=True)
    xs = range(len(train_losses))
    plt.plot(xs, train_losses, label="train loss")
    plt.plot(xs, val_losses, label="val loss")
    plt.xticks(range(0, len(train_losses), int(len(train_losses)/epochs_ae)), x_ticks)
    plt.ylabel("loss")
    plt.title("AE Train/Val Loss per Epoch")
    plt.legend()
    plt.savefig("./results/trainval_loss_ae.png")
    plt.clf()
    
    # Use encoder for downstream classification
    encoder = ae_model.encoder
    encoder.save('encoder.h5')

  if not encoder:
    encoder = tf.keras.models.load_model('encoder.h5')

  dense_model = AEClassifier(encoder, num_classes)
  train_losses = []
  val_losses = []
  x_ticks = []
  for e in range(epochs_dense):
    loss, t_losses, v_losses = train_classifier(dense_model, train_dataset, val_dataset, e)
    train_losses = train_losses + t_losses
    val_losses = val_losses + v_losses
    if e % 3 == 0:
      x_ticks.append("Epoch " + str(e))
    print("Epoch " + str(e) + ", avg loss=" + str(loss.numpy()) + "\n", flush=True)
  xs = range(len(train_losses))
  plt.plot(xs, train_losses, label="train loss")
  plt.plot(xs, val_losses, label="val loss")
  plt.xticks(range(0, len(train_losses), int(len(train_losses)/(epochs_dense/3))), x_ticks)
  plt.ylabel("loss")
  plt.title("AE Classifier Train/Val Loss per Epoch")
  plt.legend()
  plt.savefig("./results/trainval_loss_ae_classifier.png")

  # Print model metrics on train set
  confusion_matrix = test(dense_model, train_dataset)
  precision = dense_model.precision.result().numpy()
  recall = dense_model.recall.result().numpy()
  print("Train Accuracy: " + str(dense_model.categorical_accuracy.result().numpy()))
  print("Train AUROC: " + str(dense_model.auroc.result().numpy()))
  print("Train Precision: " + str(precision))
  print("Train Recall: " + str(recall))
  print("Train F1 score: " + str(2 * recall * precision / (precision + recall)))
  print("Train Confusion Matrix: " + str(confusion_matrix))    

  # Print model metrics on test set
  confusion_matrix = test(dense_model, test_dataset)
  precision = dense_model.precision.result().numpy()
  recall = dense_model.recall.result().numpy()
  print("Test Accuracy: " + str(dense_model.categorical_accuracy.result().numpy()))
  print("Test AUROC: " + str(dense_model.auroc.result().numpy()))
  print("Test Precision: " + str(precision))
  print("Test Recall: " + str(recall))
  print("Test F1 score: " + str(2 * recall * precision / (precision + recall)))
  print("Test Confusion Matrix: " + str(confusion_matrix))

  # Print model summary
  ae_model.build((None, 200, 5))
  print()
  print(ae_model.encoder.summary())
  print(ae_model.decoder.summary())
  print()
  print(dense_model.layers1.summary())


if __name__ == "__main__":
  main()

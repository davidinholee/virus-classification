import numpy as np
import tensorflow as tf
from Bio import SeqIO
import matplotlib.pyplot as plt
from main_ae import train_ae
from autoencoder_model import Autoencoder

def raw_data_analysis():
  nts = ["A", "C", "G", "T", "N"]
  virus_names = ["covid", "influenza", "mers", "sars"]
  file_names = ["./fasta_data_final/covid.fasta", 
    "./fasta_data_final/influenza.fasta",
    "./fasta_data_final/mers.fasta",
    "./fasta_data_final/sars.fasta"]
  n_samples = []
  n_complete = [0 for name in virus_names]
  n_partial = [0 for name in virus_names]
  sample_lengths = [[] for name in virus_names]
  max_length = 0

  for i in range(len(file_names)):
    virus = list(SeqIO.parse(open(file_names[i]), 'fasta'))
    n_samples.append(len(virus))
    print(virus_names[i] + ': ' + str(n_samples[i]) + ' samples')
    nt_counts = [0, 0, 0, 0, 0, 0]
    other_nts = set()
    for sample in virus:
      # Make sure COVID samples are COVID
      if virus_names[i] == "covid":
        assert(" 2 " in sample.description)
      # Make sure SARS samples are SARS
      if virus_names[i] == "sars":
        assert((" 2 " not in sample.description) or "Sequence 2" in sample.description)
        assert("SARS" or "coronavirus" in sample.description)
      # Check that there are no weird nucleotides
      for nt in sample.seq:
        if nt in nts:
          nt_counts[nts.index(nt)] += 1
        else:
          other_nts.add(nt)
          nt_counts[-1] += 1
        # assert(nt in nts)
      # Count partial and complete samples
      if "complete" in sample.description or "STRAIN" in sample.description:
        n_complete[i] += 1
      elif "partial" in sample.description or "model" in sample.description or "segment" in sample.description or "test" in sample.description or "MODIFIED" in sample.description or "COMPOSITIONS" in sample.description or "ANTIBODIES" in sample.description or "Acid" in sample.description or "gene" in sample.description or "UNVERIFIED" in sample.description or "Patent" in sample.description:
        n_partial[i] += 1
      # Record length of each sample
      sample_lengths[i].append(len(sample))
    print("Complete:", n_complete[i])
    print("Partial:", n_partial[i])
    print("Nucleotide Counts:", nt_counts, "(A, C, G, T, N, Other)")
    print("Other Nucleotides:", other_nts)
  max_length = np.max([np.max(ls) for ls in sample_lengths])
  min_length = np.min([np.min(ls) for ls in sample_lengths])
  print("\nMin Seq Length:", str(min_length), "\nMax Seq Length:", str(max_length))

  # Make charts
  fig, ax = plt.subplots()
  width = 0.35
  complete = ax.bar(virus_names, n_complete, width, label='Complete')
  partial = ax.bar(virus_names, n_partial, width, label='Partial')
  plt.bar_label(complete, label_type="edge", fontweight='bold')
  plt.bar_label(partial, label_type="center", fontweight='bold')
  ax.set_ylabel('Num Samples')
  ax.set_title('Number of Samples per Virus')
  ax.legend()
  plt.savefig("./results/n_samples.png")
  plt.clf()

  fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7,20))
  plt.subplots_adjust(hspace=0.5)
  upper_bound = round(max_length, -3)
  for i, virus in enumerate(virus_names):
    axs[i].hist(sample_lengths[i], bins=250, range=(0, upper_bound))
    axs[i].set_ylabel('Count')
    axs[i].set_title('Sequence Lengths of ' + virus)
  plt.savefig("./results/seq_lengths.png")

def nt_freq_analysis():
  nts = ["A", "C", "G", "T"]
  file_names = ["./fasta_data_final/covid.fasta", 
    "./fasta_data_final/influenza.fasta",
    "./fasta_data_final/mers.fasta",
    "./fasta_data_final/sars.fasta"]
  repeat_freqs = {}
  count = 0

  for i in range(len(file_names)):
    virus = list(SeqIO.parse(open(file_names[i]), 'fasta'))
    for sample in virus:
      repeat = 1
      prev = sample.seq[0]
      for nt in sample.seq[1:]:
        if nt == prev or (nt not in nts and prev not in nts):
          repeat += 1
        else:
          if repeat in repeat_freqs:
            repeat_freqs[repeat] += 1
          else:
            repeat_freqs[repeat] = 1
          repeat = 1
          prev = nt
          count += 1
      if repeat in repeat_freqs:
        repeat_freqs[repeat] += 1
      else:
        repeat_freqs[repeat] = 1
      count += 1
  for i in range(1, len(repeat_freqs)):
    print("Length " + str(i) + " Repeats Freq: " + str(repeat_freqs[i]/count))

def autoencoder_analysis():
  class_names = ["covid", "influenza", "mers", "sars"]
  data_path = "./fasta_data_final/dataset"
  type_spec = {"data": tf.TensorSpec(shape=[None, 5], dtype=tf.bool), 
    "class_name": tf.TensorSpec(shape=None, dtype=tf.string), 
    "class_id": tf.TensorSpec(shape=None, dtype=tf.int32)}
  train_dataset = tf.data.experimental.load(data_path + "_train", type_spec)
  val_dataset = tf.data.experimental.load(data_path + "_val", type_spec)
  test_dataset = tf.data.experimental.load(data_path + "_test", type_spec)
  for x in train_dataset:
    shape = x["data"].shape[-2:]
    break
  ae_model = Autoencoder(shape, embedding_size=2)
    
  # Train model
  train_losses = []
  val_losses = []
  x_ticks = []
  epochs = 3
  for e in range(epochs):
    loss, t_losses, v_losses = train_ae(ae_model, train_dataset, val_dataset, e)
    train_losses = train_losses + t_losses
    val_losses = val_losses + v_losses
    x_ticks.append("Epoch " + str(e))
    print("Epoch " + str(e) + ", avg loss=" + str(loss.numpy()) + "\n", flush=True)
  xs = range(len(train_losses))
  plt.plot(xs, train_losses, label="train loss")
  plt.plot(xs, val_losses, label="val loss")
  plt.xticks(range(0, len(train_losses), int(len(train_losses)/epochs)), x_ticks)
  plt.ylabel("loss")
  plt.title("AE (embedding_sz=2) Train/Val Loss per Epoch")
  plt.legend()
  plt.savefig("./results/trainval_loss_ae2.png")
  plt.clf()
  
  # Generate embeddings of training data and plot
  encoder = ae_model.encoder
  dataset = train_dataset.shuffle(16000)
  dataset = dataset.batch(400)
  embeddings = []
  labels = []
  groups_x = [[], [], [], []]
  groups_y = [[], [], [], []]
  colors = ['red', 'green', 'blue', 'purple']
  for batch in dataset:
    batch_inputs = batch["data"]
    batch_labels = batch["class_id"]
    embeds = encoder.call(tf.cast(batch_inputs, tf.float32))
    for e in embeds:
      embeddings.append(e)
    for b in batch_labels:
      labels.append(b)
    break
  for coord, label in zip(embeddings, labels):
    groups_x[label].append(coord[0])
    groups_y[label].append(coord[1])
  for i in range(4):
    plt.scatter(groups_x[i], groups_y[i], c=colors[i], label=class_names[i])
  plt.legend()
  plt.grid(True)
  plt.title("Autoencoder Embeddings by Class")
  plt.savefig("./results/embeddings.png")

nt_freq_analysis()

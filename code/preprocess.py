import numpy as np
from collections import defaultdict
import random
import tensorflow as tf
from tensorflow.data.experimental import save as tf_save
from Bio import SeqIO

def get_full_data(file_names, class_names, data_path, train_split = .8, val_split = .1, test_split = .1):
  '''
  Reads and parses a list of fasta FASTA files. Creates one_hot encoding of DNA sequence
  and tensor of labels. Sequences are split into train/test by train_test_split.
  
  :param file_names: list of paths to .fasta files, each corresponding to all sequences from one virus
  :param class_names: list of names of viruses corresponding to file_names
  :param data_path: path to save preprocessed data
  :param train_test_split: float, proportion of sequences desired to be in train set
  :return: tuple of (train_dataset, test_dataset)
  :return train_dataset: tf Dataset of training data, training labels, and class names
  :return test_dataset: tf Dataset of testing data, testing labels, and class names
  Sequences whose length is less than the longest sequence length (across all fasta files) 
  are filled with [0, 0, 0, 0, 0] encodings for dimensions to match.
  label scheme:
  file_names = [<covid.fasta>, <influenza.fasta>, <mers.fasta>, ...]
  labels covid -> 0
          influenza -> 1
          mers -> 2
          ... 
  '''

  nts = ["A", "C", "G", "T", "N"]
  sample_length = 200
  undersampling_constant = 0.9
  train_raw = []
  val_raw = []
  test_raw = []
  train_data = defaultdict(list)
  val_data = defaultdict(list)
  test_data = defaultdict(list)
  n_bps = []
  
  for i in range(len(file_names)):
    # Loads each file in
    parsed = list(SeqIO.parse(open(file_names[i]), 'fasta'))
    train_samples = int(train_split*len(parsed))
    val_samples = int(val_split*len(parsed))
    random.shuffle(parsed)
    train_raw.append(parsed[:train_samples])
    val_raw.append(parsed[train_samples:train_samples+val_samples])
    test_raw.append(parsed[train_samples+val_samples:])
    n_bps.append(np.sum([len(sample) for sample in train_raw[-1]]))
    print(file_names[i] + ': ' + str(len(parsed)) + ' samples', flush=True)
  min_bps = np.min(n_bps)

  for i in range(len(file_names)):
    # Determine whether to undersample training data (worst case this value is 1)
    accept_prob = tf.cast(np.min([min_bps/(n_bps[i] * undersampling_constant), 1]), tf.float32)
    # Create training data of seq_length segments
    class_count = 0
    for data in train_raw[i]:
      # Reject over represented classes with a certain probability
      n_samples = int(np.ceil(len(data.seq)/sample_length))
      accepts = np.random.binomial(1, accept_prob, size=n_samples)
      seq = str(data.seq)
      for nt in ["M"]:
        seq = seq.replace(nt, "N")
      for j in range(n_samples):
        # We don't care about the leftover bit at the end if the sequence is long
        if accepts[j] and (j < n_samples - 1 or j < 5):
          if j < n_samples - 1:
            data_str = seq[j*sample_length:(j+1)*sample_length]
          else:
            data_str = seq[j*sample_length:] + (' ' * (sample_length - len(seq[j*sample_length:])))
          data_num = tf.stack([tf.equal(list(data_str), nt) for nt in nts], axis=-1)
          train_data['data'].append(data_num)
          train_data['class_name'].append(class_names[i])
          train_data['class_id'].append(i)
          class_count += 1
    print("Training data class " + class_names[i] + " count: " + str(class_count))
    
    # Create testing data
    for data in val_raw[i]:
      # Reject over represented classes with a certain probability
      n_samples = int(np.ceil(len(data.seq)/sample_length))
      # If you don't want to exclude any val data, uncomment
      accepts = np.random.binomial(1, accept_prob, size=n_samples) # [1 for _ in range(n_samples)]
      seq = str(data.seq)
      for nt in ["M"]:
        seq = seq.replace(nt, "N")
      for j in range(n_samples):
        # We don't care about the leftover bit at the end if the sequence is long
        if accepts[j] and (j < n_samples - 1 or j < 5):
          if j < n_samples - 1:
            data_str = seq[j*sample_length:(j+1)*sample_length]
          else:
            data_str = seq[j*sample_length:] + (' ' * (sample_length - len(seq[j*sample_length:])))
          data_num = tf.stack([tf.equal(list(data_str), nt) for nt in nts], axis=-1)
          val_data['data'].append(data_num)
          val_data['class_name'].append(class_names[i])
          val_data['class_id'].append(i)

    # Create testing data
    for data in test_raw[i]:
      # Reject over represented classes with a certain probability
      n_samples = int(np.ceil(len(data.seq)/sample_length))
      # If you don't want to exclude any test data, uncomment
      accepts = np.random.binomial(1, accept_prob, size=n_samples) # [1 for _ in range(n_samples)]
      seq = str(data.seq)
      for nt in ["M"]:
        seq = seq.replace(nt, "N")
      for j in range(n_samples):
        # We don't care about the leftover bit at the end if the sequence is long
        if accepts[j] and (j < n_samples - 1 or (j < 5 and len(seq[j*sample_length:]) > 0.5 * sample_length)):
          if j < n_samples - 1:
            data_str = seq[j*sample_length:(j+1)*sample_length]
          else:
            data_str = seq[j*sample_length:] + (' ' * (sample_length - len(seq[j*sample_length:])))
          data_num = tf.stack([tf.equal(list(data_str), nt) for nt in nts], axis=-1)
          test_data['data'].append(data_num)
          test_data['class_name'].append(class_names[i])
          test_data['class_id'].append(i)
  print("Training data shape:", len(train_data['data']), train_data['data'][0].shape, flush=True)
  print("Validation data shape:", len(val_data['data']), val_data['data'][0].shape, flush=True)
  print("Testing data shape:", len(test_data['data']), test_data['data'][0].shape, flush=True)
  print()
  
  # Convert to tf Dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(dict(train_data))
  train_dataset = train_dataset.shuffle(len(train_data['data']))
  val_dataset = tf.data.Dataset.from_tensor_slices(dict(val_data))
  val_dataset = val_dataset.shuffle(len(val_data['data']))
  test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_data))
  test_dataset = test_dataset.shuffle(len(test_data['data']))
  tf_save(train_dataset, data_path + "_train")
  tf_save(val_dataset, data_path + "_val")
  tf_save(test_dataset, data_path + "_test")

  return train_dataset, val_dataset, test_dataset

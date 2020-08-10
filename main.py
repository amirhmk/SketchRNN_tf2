import os
import sys
import math
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from tensorflow import keras as K

print("tf: {}".format(tf.version.VERSION))
print("tf.keras: {}".format(K.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from sketchrnn import models, dataset, utils
data_class = 'rabbit' #@param ["cat","eye","rabbit"]
data = np.load(f'{data_class}.npz',encoding='latin1',allow_pickle=True)
data_train = [dataset.cleanup(d) for d in data['train']]
data_valid = [dataset.cleanup(d) for d in data['valid']]
data_test = [dataset.cleanup(d) for d in data['test']]

hps = {
    "max_seq_len": max(map(len, np.concatenate([data['train'], data['valid'], data['test']]))),
    'batch_size': 100,
    "num_batches": math.ceil(len(data_train) / 100),
    "epochs": 100,
    "recurrent_dropout_prob": 0.1, ## 0.0 for gpu lstm
    "enc_rnn_size": 256,
    "dec_rnn_size": 512,
    "z_size": 128,
    "num_mixture": 20,
    "learning_rate": 0.001,
    "min_learning_rate": 0.00001,
    "decay_rate": 0.9999,
    "grad_clip": 1.0,
    'kl_tolerance': 0.2,
    'kl_decay_rate': 0.99995,
    "kl_weight": 0.5,
    'kl_weight_start': 0.01,
}

sketchrnn = models.SketchRNN(hps)
sketchrnn.models['full'].summary()

scale_factor = dataset.calc_scale_factor(data_train)

train_dataset = dataset.make_train_dataset(data_train, hps['max_seq_len'], hps['batch_size'], scale_factor)
val_dataset = dataset.make_val_dataset(data_valid, hps['max_seq_len'], hps['batch_size'], scale_factor)

checkpoint_dir = 'weights/'
log_dir = 'log/'


initial_epoch = 0 #@param {type: "number"}
initial_loss = 0.05 #@param {type: "number"}
checkpoint = os.path.join(checkpoint_dir, 'sketch_rnn_' + data_class + '_weights.{:02d}_{:.2f}.hdf5')

if initial_epoch > 0:
    sketchrnn.load_weights(checkpoint.format(initial_epoch, initial_loss))

sketchrnn.train(initial_epoch, train_dataset, val_dataset, checkpoint)

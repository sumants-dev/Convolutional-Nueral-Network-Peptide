
import jax.numpy as np
from jax import  nn
from jax import random
from jax.experimental import stax
from jax.experimental.stax import BatchNorm, MaxPool, Conv, Dense, Flatten, Relu, Softmax, Sigmoid, normalize, Dropout, SumPool, FanOut
from jax.experimental import optimizers
from jax import jit, grad, vmap, jvp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from jax.ops import index_update, index
import numpy as onp
from tqdm import tqdm
import itertools
from functools import partial
import time
from tqdm import trange
import numpy.random as npr
from jax import vjp
from jax import grad, vmap
import pandas as pd




def processSeqData():
  df = pd.read_csv('amyloid_db.csv')
  peptides = df['peptide'].values
  results = np.array(list(df['Binary_Classification'].values))
  unique_AA = onp.array([onp.array(list(peptide)) for peptide in peptides])
  uniq_map = []
  for AAs in unique_AA:
    for AA in AAs:
      uniq_map.append(AA)

  keys = sorted(list(set(uniq_map)))
  values = np.arange(1, len(keys)+1)
  AAToNum = {key: value for key, value in zip(keys, values)}

  XTrain = []
  targets = []
  idx = 0
  idxs = []
  for peptide, result in zip(peptides, results):
    sample = []

    for aa in list(peptide):
      sample.append(AAToNum[aa])
    if(len(sample) == 6):
      XTrain.append(np.array(sample))
      idxs.append(peptide)
      targets.append(result)
    idx+=1


  XTrain = np.array(XTrain)
  XTrain = nn.one_hot(XTrain, len(keys))
  X = np.expand_dims(XTrain, 3)
  
  train, test = onp.load('Training_Peptide_Set.npy', allow_pickle = True), onp.load('Testing_Peptide_Set.npy', allow_pickle = True)

  msk_train = []
  msk_test = []

  for i in idxs:
    if (i in train):
      msk_train.append(True)
    else:
      msk_train.append(False)

    if (i in test):
      msk_test.append(True)
    else:
      msk_test.append(False)
  msk_train, msk_test = np.array(msk_train), np.array(msk_test)
  targets = np.array(targets)
  return X[msk_train], targets[msk_train], X[msk_test], targets[msk_test]

# Architecture
def make_net(mode = 'train'):
  init_fun, conv_net = stax.serial(
    Conv(out_chan=20, filter_shape=(5, 5), strides=(1, 1), padding="SAME"),
    BatchNorm(),
    Relu,
    MaxPool((2, 2), strides= (1, 1), padding="SAME"),
    Conv(out_chan=12, filter_shape=(5, 5), strides=(1, 1), padding="SAME"),
    BatchNorm(), 
    Relu,
    SumPool((2,2), strides=(2,2), padding="SAME"),
    Conv(out_chan=12, filter_shape=(5, 5), strides=(1, 1), padding="SAME"),
    BatchNorm(),
    Relu,
    Flatten, 
    Dense(256),
    Relu,
    Dropout(.3, mode = mode),                                
    Dense(128),
    Relu,
    Dense(64),
    Relu,
    Dense(2),
    Softmax)

  return init_fun, conv_net


lr = optimizers.exponential_decay(1e-3, decay_rate = 0.99, decay_steps = 500)
opt_init, opt_update, get_params = optimizers.adam(lr)
key = random.PRNGKey(1)
key, subkey = random.split(key)

# Cross entropy, y is index of label NOT one hot.
def loss(params, batch):
    X, y = batch
    _ , conv_net = make_net()
    return -np.log(conv_net(params, X, rng = key)[np.arange(y.shape[0]), y]).mean()

def accuracy(params, batch):
    X, y = batch
    _ , conv_net = make_net(mode = 'test')
    y_pred = np.argmax(conv_net(params, X, rng = key),1)

    return (y_pred == y).sum() / X.shape[0]

@jit
def step(i, opt_state, batch):
    params = get_params(opt_state)
    g = grad(loss)(params, batch)

    return opt_update(i, g, opt_state)

def train(x_train, y_train, x_test, y_test, num_epoch, batch_size, key = random.PRNGKey(1)):
  init_fun, conv_net = make_net()
  _, init_params = init_fun(key, x_train[0].shape)
  opt_state = opt_init(init_params)
  epoch_key = random.split(key, num_epoch)

  nIter = x_train[0].shape[0] // batch_size
  for i in range(num_epoch):
      perm_idx = random.permutation(epoch_key[i], np.arange(x_train[0].shape[0]))
      epoch_x_data = x_train[perm_idx]
      epoch_y_data = y_train[perm_idx]

      for j in range(nIter):
          batch_x_data = epoch_x_data[j * batch_size : (j + 1) * batch_size]
          batch_y_data = epoch_y_data[j * batch_size : (j + 1) * batch_size]
          opt_state = step(i * nIter + j + 1, opt_state, (batch_x_data, batch_y_data))

      params = get_params(opt_state)
      print('Epoch : ', i + 1, 'Training acc. : ', accuracy(params, (x_train, y_train)),'Testing acc  :', accuracy(params, (x_test, y_test)))
  return conv_net, params

def getXData(enrFile):
  idxs = np.load('Index.npy')
  train, test = onp.load('Training_Peptide_Set.npy', allow_pickle = True), onp.load('Testing_Peptide_Set.npy', allow_pickle = True)
  msk_train, msk_test = [], []
  for i in idxs:
    if (i in train):
      msk_train.append(True)
    else:
      msk_train.append(False)

    if (i in test):
      msk_test.append(True)
    else:
      msk_test.append(False)
  msk_train, msk_test = np.array(msk_train), np.array(msk_test)

  XTrain = np.load(enrFile)
  X = np.expand_dims(XTrain, 3)
  #X = nn.normalize(X)
  mu, sigma = np.mean(X), np.std(X)

  X = (X - mu) /sigma
  
  train, test = onp.load('Training_Peptide_Set.npy', allow_pickle = True), onp.load('Testing_Peptide_Set.npy', allow_pickle = True)
  
  y_train_idx = idxs[msk_train]
  y_test_idx = idxs[msk_test]
  df = pd.read_csv('amyloid_db.csv')
  df = df.set_index('peptide')
  y_train = np.array(list(df.loc[y_train_idx].values))
  y_test = np.array(list(df.loc[y_test_idx].values))

  y_train, y_test = y_train.flatten(), y_test.flatten()
  return X[msk_train], y_train, X[msk_test], y_test


fa_elec_x_train, fa_elec_y_train, fa_elec_x_test, fa_elec_y_test = getXData('Feature_fa_elec.npy') 
fa_atr_x_train, fa_atr_y_train, fa_atr_x_test, fa_atr_y_test = getXData('Feature_fa_atr.npy') 
fa_rep_x_train, fa_rep_y_train, fa_rep_x_test, fa_rep_y_test = getXData('Feature_fa_rep.npy')



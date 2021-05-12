from numpy.core.fromnumeric import shape
from jax import numpy as np
from jax.nn import sigmoid, log_sigmoid, normalize
from jax import random
from jax import random, jit, grad, vmap
from jax.experimental import optimizers
import numpy as onp
import pandas as pd

# Performance Metrics
def accuracy(y_pred, y):
    return (y_pred == y).sum() / len(y)

# Setting Up Logistic Regression in Jax

## Optimizer Set up with default options
lr = optimizers.exponential_decay(1e-3, decay_rate = 0.99, decay_steps = 500)
opt_init, opt_update, get_params = optimizers.adam(lr)

## Basis Functions
def identityBasis(x):
  return x

def logistic(r, basis = identityBasis ):
  return 1 / (1 + np.exp(-basis(r)))

def predict(params, X):
  c, w = params
  p = logistic(np.dot(X, w) + c)

  return (p >= .5).astype(int)

def predictProba(params, X):
  c, w = params
  return logistic(np.dot(X, w) + c)

def loss(params, batch, eps= 1e-14):
  X, y = batch
  n = len(y)
  p = predictProba(params, X)
  p = np.clip(p, eps, 1-eps)

  return -np.sum((y*np.log(p)) + (1-y)*np.log(1-p))

@jit
def step(i, opt_state, batch):
    params = get_params(opt_state)
    g = grad(loss)(params, batch)

    return opt_update(i, g, opt_state)

def trainModel(x_train, y_train, x_test, y_test, num_epoch = 1000, batch_size = 128, key = random.PRNGKey(1), tol = 1e-4):
  
  w = random.uniform(key, shape = (x_test.shape[1],))
  c = random.uniform(key, shape = (1,))
  init_params = (c[0], w)
  
  opt_state = opt_init(init_params)
  epoch_key = random.split(key, num_epoch)

  losses, accTest, accTrain = [], [], []
  nIter = x_train.shape[0] // batch_size
  for i in range(num_epoch):
      perm_idx = random.permutation(epoch_key[i], np.arange(x_train.shape[0]))
      epoch_x_data = x_train[perm_idx]
      epoch_y_data = y_train[perm_idx]
      
      for j in range(nIter):
        batch_x_data = epoch_x_data[j * batch_size : (j + 1) * batch_size]
        batch_y_data = epoch_y_data[j * batch_size : (j + 1) * batch_size]
        opt_state = step(i * nIter + j + 1, opt_state, (batch_x_data, batch_y_data))

      params = get_params(opt_state)
      losses.append(loss(params, (x_train, y_train)))
      print('Epoch '+ str(i) +'\tTrain Acc: ' + str(accuracy(predict(params, x_train), y_train)) + '\tTest Acc: ' + str(accuracy(predict(params, x_test), y_test)) + '\tLoss: '+ str(losses[-1]))
      if (i > 20) and (i % 10 == 0):
        if np.abs(losses[-1] - losses[-20]) < tol:
            print(f"Exited loop at iteration {i}")
            break

  return params



df = pd.read_csv('amyloid_db_foldx_data.csv')  
df = df.set_index('Sequence')

train, test = onp.load('Training_Peptide_Set.npy', allow_pickle = True), onp.load('Testing_Peptide_Set.npy', allow_pickle = True)
df_train, df_test = df.loc[train], df.loc[test]

X_train = normalize(np.array(list(df_train[df_train.columns[:-1]].values)))
y_train = np.array(list(df_train[df_train.columns[-1]].values))


X_test = normalize(np.array(list(df_test[df_test.columns[:-1]].values)))
y_test = np.array(list(df_test[df_test.columns[-1]].values))

params = trainModel(X_train, y_train, X_test, y_test)


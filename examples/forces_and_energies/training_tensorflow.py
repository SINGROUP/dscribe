import numpy as np
import tensorflow as tf
import ase
from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

import time
import math

# Load the dataset
D_numpy = np.load("D.npy")[:, 0, :]  # We only have one SOAP center
n_samples, n_features = D_numpy.shape
E_numpy = np.array([np.load("E.npy")]).T
F_numpy = np.load("F.npy")
dD_dr_numpy = np.load("dD_dr.npy")[:, 0, :, :, :]  # We only have one SOAP center
r_numpy = np.load("r.npy")

tf.random.set_seed(2)

# Select equally spaced points for training
n_train = 30
idx = np.linspace(0, len(r_numpy) - 1, n_train).astype(int)

D_train_full = D_numpy[idx]
E_train_full = E_numpy[idx]
F_train_full = F_numpy[idx]
dD_dr_train_full = dD_dr_numpy[idx]
r_train_full = r_numpy[idx]


scaler = StandardScaler().fit(D_train_full)
D_train_full = scaler.transform(D_train_full)
D_whole = scaler.transform(D_numpy)

dD_dr_train_full = dD_dr_train_full / scaler.scale_[None, None, None, :]
dD_dr_whole = dD_dr_numpy / scaler.scale_[None, None, None, :]

# Calculate the variance of energy and force values for the training set. These
# are used to balance their contribution to the MSE loss
var_force_train = F_train_full.var()
var_energy_train = E_train_full.var()

# Subselect 20% of validation points for early stopping.
D_train, D_valid, E_train, E_valid, F_train, F_valid, dD_dr_train, dD_dr_valid, r_train, r_valid = train_test_split(
    D_train_full,
    E_train_full,
    F_train_full,
    dD_dr_train_full,
    r_train_full,
    test_size=0.2,
    random_state=7
)


# Create model
def make_model(n_features, n_hidden, n_out):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, input_dim=n_features, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(n_out, activation='linear'))
    return model


# Loss function: decorated with @tf.function decorator to create computation
# graphs for different parts of the training step, abstracting certain parts of
# the training steps allows to speed up training time compared to using simple
# Eager Execution which is Default for tensorflow 2.0
@tf.function
def force_energy_loss(E_hat, F_hat, E, F):
    E_loss = tf.math.reduce_mean((E_hat - E)**2) / var_energy_train
    F_loss = tf.math.reduce_mean((F_hat - F)**2) / var_force_train
    return E_loss + F_loss


# Create model with 1 hidden layer and 5 nodes
model = make_model(n_features, 5, 1)
opt = tf.keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-8)

n_max_epochs = 5000
batch_size = 2
patience = 20
i_worse = 0

old_valid_loss = float("Inf")
best_valid_loss = float("Inf")

# Create Tensors and cast to float32 to avoid type errors
D_train_tf = tf.constant(D_train)
D_val_tf = tf.constant(D_valid)
D_train_tf = tf.cast(D_train_tf, dtype=tf.float32)
D_val_tf = tf.cast(D_val_tf, dtype=tf.float32)

E_train_tf = tf.constant(E_train)
E_val_tf = tf.constant(E_valid)
E_train_tf = tf.cast(E_train_tf, dtype=tf.float32)
E_val_tf = tf.cast(E_val_tf, dtype=tf.float32)

F_train_tf = tf.constant(F_train)
F_val_tf = tf.constant(F_valid)
F_train_tf = tf.cast(F_train_tf, dtype=tf.float32)
F_val_tf = tf.cast(F_val_tf, dtype=tf.float32)

dD_dr_train_tf, dD_dr_val_tf = tf.constant(dD_dr_train), tf.constant(dD_dr_valid)
dD_dr_train_tf = tf.cast(dD_dr_train_tf, dtype=tf.float32)
dD_dr_val_tf = tf.cast(dD_dr_val_tf, dtype=tf.float32)


perm = [i for i in range(len(D_train_tf))]
perm = tf.random.shuffle(perm)


# Initialises nested GradientTapes to implement forward pass then back
# propagation. Once again decorated with @tf.function for speedup.
@tf.function
def calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(D_train_batch)
            E_pred = model(D_train_batch, training=True)
            df_dD_train_batch = tape2.gradient(E_pred, D_train_batch)
        F_pred = -tf.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)
        loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)

        grads = tape1.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))


# Calculates validation loss
@tf.function
def calc_val_loss(E_val_tf, F_val_tf, D_val_tf):
    with tf.GradientTape() as val_tape:
        val_tape.watch(D_val_tf)
        E_pred_val = model(D_val_tf, training=False)

    df_dD_val = val_tape.gradient(E_pred_val, D_val_tf)
    F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val_tf, df_dD_val)

    return force_energy_loss(E_pred_val, F_pred_val, E_val_tf, F_val_tf)


# Implements one full training step for current batch
@tf.function
def step(indices):
    D_train_batch, E_train_batch = tf.gather(D_train_tf, indices), tf.gather(E_train_tf, indices)
    F_train_batch, dD_dr_train_batch = tf.gather(F_train_tf, indices), tf.gather(dD_dr_train_tf, indices)
    calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt)


# Trains model invoking helper functions (equivalent to model.fit())
def train_model(perm, best_valid_loss, old_valid_loss, i_worse):
    for i in range(n_max_epochs):

        perm = tf.random.shuffle(perm)
        for j in range(0, len(D_train_tf), batch_size):
            lst = [q for q in range(j, j + batch_size)]
            indices = tf.gather(perm, lst).numpy()
            step(indices)

        # Validation stage
        val_loss = calc_val_loss(E_val_tf, F_val_tf, D_val_tf)
        if val_loss < best_valid_loss:
            model.save('model')
            best_valid_loss = val_loss
        if val_loss >= old_valid_loss:
            i_worse += 1
        else:
            i_worse = 0
        if i_worse > patience:
            tf.print("Early stopping at Epoch {}".format(i))
            break
        old_valid_loss = val_loss
        if i % 500 == 0:
            print("Finished epoch: {} with val_loss: {}".format(i, val_loss))


train_model(perm, best_valid_loss, old_valid_loss, i_worse)

# Enter evaluation stage: load best model
final_model = load_model('model')

# Create tf.tensors for the whole spaces
D_tf = tf.constant(D_whole)
D_tf = tf.cast(D_tf, tf.float32)

F_tf = tf.constant(F_numpy)
F_tf = tf.cast(F_tf, tf.float32)

E_tf = tf.constant(E_numpy)
E_tf = tf.cast(E_tf, tf.float32)

dD_dr_tf = tf.constant(dD_dr_whole)
dD_dr_tf = tf.cast(dD_dr_tf, tf.float32)

with tf.GradientTape() as test_tape:
    test_tape.watch(D_tf)
    E_whole_pred = final_model(D_tf, training=False)

df_dD_whole = test_tape.gradient(E_whole_pred, D_tf)
F_whole_pred = -tf.einsum('ijkl,il->ijk', dD_dr_tf, df_dD_whole)

F_whole = F_tf.numpy()
E_whole = E_tf.numpy()

E_whole_pred = E_whole_pred.numpy()
F_whole_pred = F_whole_pred.numpy()

# Save results for later analysis
np.save("r_train_full.npy", r_train_full)
np.save("E_train_full.npy", E_train_full)
np.save("F_train_full.npy", F_train_full)
np.save("E_whole_pred.npy", E_whole_pred)
np.save("F_whole_pred.npy", F_whole_pred)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:58:39 2021

@author: scottleroux
"""

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

#load the datasets
path = ".."
D_numpy = np.load(f"{path}/D.npy")[:, 0, :]  # We only have one SOAP center
n_samples, n_features = D_numpy.shape
E_numpy = np.array([np.load(f"{path}/E.npy")]).T
F_numpy = np.load(f"{path}/F.npy")
dD_dr_numpy = np.load(f"{path}/dD_dr.npy")[:,0,:,:,:] # We only have one SOAP center 
r_numpy = np.load(f"{path}/r.npy")

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
dD_dr_whole = dD_dr_numpy / scaler.scale_[None,None, None, :]

#calculate variance of Energey and Forces in training set
var_force_train = F_train_full.var()
var_energy_train = E_train_full.var()


#split training into training and validation 
D_train, D_valid, E_train, E_valid, F_train, F_valid, dD_dr_train, dD_dr_valid, r_train, r_valid = train_test_split(
    D_train_full,
    E_train_full,
    F_train_full,
    dD_dr_train_full,
    r_train_full,
    test_size=.2,
    random_state = 7)

#TensorFlow code Start

#create model 
def make_model(n_features, n_hidden, n_out):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, input_dim = n_features, activation = 'sigmoid'))
    model.add(tf.keras.layers.Dense(n_out, activation = 'linear'))
    return model

#loss function: decorated with @tf.function to increase speed
@tf.function
def force_energy_loss(E_hat, F_hat, E, F):
    E_loss = tf.math.reduce_mean((E_hat - E)**2) / var_energy_train
    F_loss = tf.math.reduce_mean((F_hat - F)**2) / var_force_train
    return E_loss + F_loss

#create model with 1 hidden layer and 5 nodes
model = make_model(n_features, 5, 1)
opt = tf.keras.optimizers.Adam(learning_rate = 1e-2, epsilon = 1e-8)


"""
|the difference between this and the normal training.py version        |
|is that we need to use @tf.function decorator to create computation   |
|graphs for different parts of the training step, abstracting certain  |
|parts of the training steps allows to speed up training time compared |
|to using simple Eager Execution which is Default for tensorflow 2.0   |

Note: The two files are identical in terms of logic and computation, simply organising the code in an alternative way to avail of Computation Graphs
		see training.py for annotation on logic and code.
"""

n_max_epochs = 5000
batch_size = 2
patience = 20
i_worse = 0

old_valid_loss = float("Inf")
best_valid_loss = float("Inf")

#create Tensors and cast to float32 to avoid type errors
D_train_tf = tf.constant(D_train)
D_val_tf = tf.constant(D_valid)
D_train_tf = tf.cast(D_train_tf, dtype = tf.float32)
D_val_tf = tf.cast(D_val_tf, dtype = tf.float32)

E_train_tf = tf.constant(E_train)
E_val_tf = tf.constant(E_valid)
E_train_tf = tf.cast(E_train_tf, dtype = tf.float32)
E_val_tf = tf.cast(E_val_tf, dtype = tf.float32)

F_train_tf = tf.constant(F_train)
F_val_tf = tf.constant(F_valid)
F_train_tf = tf.cast(F_train_tf, dtype = tf.float32)
F_val_tf = tf.cast(F_val_tf, dtype = tf.float32)

dD_dr_train_tf, dD_dr_val_tf = tf.constant(dD_dr_train), tf.constant(dD_dr_valid)
dD_dr_train_tf = tf.cast(dD_dr_train_tf, dtype = tf.float32)
dD_dr_val_tf = tf.cast(dD_dr_val_tf, dtype = tf.float32)


perm = [i for i in range(len(D_train_tf))]
perm = tf.random.shuffle(perm)


#Initialises nested GradientTapes to implement forward pass then back propogation
@tf.function
def calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt):
     with tf.GradientTape() as tape1:
          with tf.GradientTape() as tape2:
              tape2.watch(D_train_batch)
              E_pred = model(D_train_batch, training=True)        
          
		  df_dD_train_batch = tape2.gradient(E_pred, D_train_batch) 
          F_pred = -tf.einsum('ijkl,il->ijk',  dD_dr_train_batch, df_dD_train_batch)
          loss = force_energy_loss(E_pred, F_pred, E_train_batch, F_train_batch)
     
     grads = tape1.gradient(loss, model.trainable_variables)
     opt.apply_gradients(zip(grads, model.trainable_variables))
    

#calculates validation loss
@tf.function
def calc_val_loss(E_val_tf, F_val_tf, D_val_tf):
    with tf.GradientTape() as val_tape:
        val_tape.watch(D_val_tf)
        E_pred_val = model(D_val_tf, training=False)
                 
    df_dD_val = val_tape.gradient(E_pred_val, D_val_tf)       
    F_pred_val = -tf.einsum('ijkl,il->ijk', dD_dr_val_tf, df_dD_val)
    
	return force_energy_loss(E_pred_val, F_pred_val, E_val_tf, F_val_tf)      


#implements one full training step for current batch
@tf.function
def step(indices):
    D_train_batch, E_train_batch = tf.gather(D_train_tf,indices), tf.gather(E_train_tf,indices)
    F_train_batch, dD_dr_train_batch = tf.gather(F_train_tf, indices), tf.gather(dD_dr_train_tf, indices)
    calc_gradients(D_train_batch, E_train_batch, F_train_batch, dD_dr_train_batch, opt)


#trains model invoking helper functions (equivalent to model.fit())
def train_model(perm, best_valid_loss, old_valid_loss, i_worse):
    for i in range(n_max_epochs):
        
		perm = tf.random.shuffle(perm)
        for j in range(0,len(D_train_tf), batch_size):
            lst = [q for q in range(j,j+batch_size)]
            indices = tf.gather(perm, lst).numpy()
            step(indices)
        
		#validation stage
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

"""
identical as training.py for testing and plotting
"""

#enter evaluation stage
#load best model 
final_model = load_model('model')

#create tf.tensors for the whole spaces
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

r_whole = r_numpy 

order = np.argsort(r_whole)


# Plot energies for the whole range
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,10))
ax1.plot(r_whole[order], E_whole[order], label = "True", linewidth = 3, linestyle = '-')
ax1.plot(r_whole[order], E_whole_pred[order], label = "Predicted", linewidth=3, linestyle="-")
ax1.set_ylabel("Energy", size = 15)
mae_energy = mean_absolute_error(E_whole,E_whole_pred)
ax1.text(0.95, .5, "MAE: {:.2} eV".format(mae_energy),  
         horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)

# Plot forces for whole range
F_x_whole_pred = F_whole_pred[order,0,0]
F_x_whole = F_whole[:,0,0][order]
ax2.plot(r_whole[order], F_x_whole[order], label = "True", linewidth = 3, linestyle = '-')
ax2.plot(r_whole[order], F_x_whole_pred[order], label = "Predicted", linewidth = 3, linestyle = '-')
ax2.set_ylabel("Force", size = 15)
mae_force = mean_absolute_error(F_x_whole,F_x_whole_pred)
ax2.text(0.95, .5, "MAE: {:.2} eV/Å".format(mae_force),  
         horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)

#Plot training points
F_train = F_train[:, 0, 0]
ax1.scatter(r_train, E_train, marker="o", color="k", s=20, label="Training points", zorder=3)
ax2.scatter(r_train, F_train, marker="o", color="k", s=20, label="Training points", zorder=3)

#Show plot
ax1.legend(fontsize=12)
plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, hspace=0)
plt.show()








        

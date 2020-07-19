import numpy as np
import os

a = np.load('tesseral_mat_nopow.npy')
if os.path.isfile("thisIsTheEndOfAllHope.txt"):
    os.remove("thisIsTheEndOfAllHope.txt")
print(a.shape)
with open('thisIsTheEndOfAllHope.txt','a') as f:
    for i in range(a.shape[0]):
        for j in range(2*i + 1):
            print(a[i,j],  file=f)

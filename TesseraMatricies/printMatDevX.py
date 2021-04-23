import numpy as np
import os

a = np.load('tesseral_mat_nopowDevZ.npy')
if os.path.isfile("finalSoapFunctionsWithoutSqrtPi3DevZ.txt"):
    os.remove("finalSoapFunctionsWithoutSqrtPi3DevZ.txt")
print(a.shape)
with open('finalSoapFunctionsWithoutSqrtPi3DevZ.txt','a') as f:
    for i in range(a.shape[0]):
        for j in range(2*i + 1):
            print(a[i,j],  file=f)

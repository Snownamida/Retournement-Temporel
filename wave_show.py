import numpy as np

from paramètres import *

u = np.load("./wave/" + para_string + ".npy")

def haha():
    print(np.max(np.abs(u)))
    print(np.max(u))
    print(np.min(u))

    print(np.max(np.abs(u[100:])))
    print(np.max(u[100:]))
    print(np.min(u[100:]))

print(np.where(np.max(u[:,0,:],1)!=0)[0][0])

# print(u[75,0,Ny//2-5:Ny//2+5])

print(np.max(np.abs(u[100:])))
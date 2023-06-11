import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import time

from paramètres import *


u = np.load("./wave/" + para_string + ".npz")["u"]
fig,ax=plt.subplots()
ax.scatter(*(np.argwhere(coeur).T*dl),c="r",s=1)
fig.show()
plt.pause(0)
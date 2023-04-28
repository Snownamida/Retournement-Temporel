import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import time

from paramètres import *


u = np.load("./wave/" + para_string + ".npz")["u"]
fig,ax=plt.subplots()
ax.plot(u[:,56,158])
fig.show()
plt.pause(0)
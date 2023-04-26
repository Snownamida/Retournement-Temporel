import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from paramètres import *

u = np.load("./wave/" + para_string + ".npy")

fig, ax = plt.subplots(figsize=(16, 9))
u_max =np.max(np.abs(u[100:]))

def animate(i):
    ax.clear()
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t={i*dt:.5f}")
    ax.imshow(u[i].T, cmap="coolwarm", vmin=-u_max, vmax=u_max, extent=[0, Lx, 0, Ly])
    return ax


anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=50)
anim.save("./wave/" + para_string + ".mp4", writer="ffmpeg", fps=60)

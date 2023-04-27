import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from paramètres import *

u = np.load("./wave/" + para_string + ".npz")['u']

fig, ax = plt.subplots(figsize=(16, 9))
u_max =0.5


print('rendering...')
t0=time.time()

def animate(n):
    global t0
    if not n%10 :
        t1=time.time()
        print(f"\r{n}/{Nt} le temps reste estimé : {(Nt-n)*(t1-t0)/10:.2f} s",end='',flush=True)
        t0=t1
    ax.clear()
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t={n*dt:.5f}")
    ax.imshow(u[n].T, cmap="coolwarm", vmin=-u_max, vmax=u_max, extent=[0, Lx, 0, Ly])
    return ax

anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=50)
anim.save("./wave/" + para_string + ".mp4", writer="ffmpeg", fps=60)
print('\ndone')
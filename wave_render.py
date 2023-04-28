import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from paramètres import *

u = np.load("./wave/" + para_string + ".npz")["u"]


fps = 40
render_time = T  # temps de rendu
render_speed = 0.1
# 1 seconde du temps réel correspond à combien seconde du temps de rendu

N_frame = int(fps * render_time / render_speed)

fig, ax = plt.subplots(figsize=(16, 9))
u_max = 0.1

print("rendering...")
t0 = time.time()


def animate(n_frame):
    global t0
    n = int(render_speed / dt / fps * n_frame)
    if not n_frame % 10:
        t1 = time.time()
        print(
            f"\r{n_frame}/{N_frame} le temps reste estimé : {(N_frame-n_frame)*(t1-t0)/10:.2f} s",
            end="",
            flush=True,
        )
        t0 = t1
    ax.clear()
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t={n*dt:.5f}")
    u_img = ax.imshow(
        u[n, :, ::-1].T, cmap="coolwarm", vmin=-u_max, vmax=u_max, extent=[0, Lx, 0, Ly]
    )
    coeur_img=ax.scatter(*(np.argwhere(coeur).T*dl),c="r",s=1)

    return u_img, coeur_img


anim = animation.FuncAnimation(fig, animate, frames=N_frame, interval=50, blit=True)
anim.save("./wave/" + para_string + ".mp4", writer="ffmpeg", fps=60)
print("\ndone")

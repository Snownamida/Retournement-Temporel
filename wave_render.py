import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from paramètres import *

u = np.load("./wave/" + para_string + ".npz")["u"]


fps = 40
render_time = T  # temps de rendu
render_speed = 0.15
# 1 seconde du temps réel correspond à combien seconde du temps de rendu

N_frame = int(fps * render_time / render_speed)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim([0, Lx])
ax.set_ylim([0, Ly])
ax.set_xlabel("x")
ax.set_ylabel("y")
u_max = 0.1

print("rendering...")
t0 = time.time()


u_img = ax.imshow([[]], cmap="seismic", vmin=-u_max, vmax=u_max, extent=[0, Lx, 0, Ly],zorder=0)
coeur_img = ax.scatter([], [], c="r", s=1,zorder=5)


def animate(n_frame):
    global t0
    n = int(render_speed / dt / fps * n_frame)
    ax.set_title(f"t={n*dt:.5f}")
    u_img.set_data(u[n, :, ::-1].T)
    coeur_img.set_offsets(np.argwhere(coeur) * dl)
    if not n_frame % 10:
        t1 = time.time()
        print(
            f"\r{n_frame}/{N_frame} le temps reste estimé : {(N_frame-n_frame)*(t1-t0)/10:.2f} s",
            end="",
            flush=True,
        )
        t0 = t1

    return u_img, coeur_img


anim = animation.FuncAnimation(fig, animate, frames=N_frame, interval=50, blit=True)
anim.save("./wave/" + para_string + ".mp4", writer="ffmpeg", fps=fps)
print("\ndone")

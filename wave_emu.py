import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nx = 80
Ny = 80

x_n = np.linspace(0, 16, Nx)
y_n = np.linspace(0, 9, Ny)
dx, dy = x_n[1] - x_n[0], y_n[1] - y_n[0]

T = 100
Nt = 2000
dt = T / Nt

c = 1  # wave speed
u = np.zeros([Nt, len(x_n), len(y_n)])

u[0, Nx // 2, Ny // 2] = np.sin(0)
u[1, Nx // 2, Ny // 2] = np.sin(1 / 10)

for t in range(1, Nt - 1):
    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            if t < 100:
                u[t, Nx // 2, Ny // 2] = np.sin(t / 10)

            u[t + 1, x, y] = (
                c**2
                * dt**2
                * (
                    ((u[t, x + 1, y] - 2 * u[t, x, y] + u[t, x - 1, y]) / (dx**2))
                    + ((u[t, x, y + 1] - 2 * u[t, x, y] + u[t, x, y - 1]) / (dy**2))
                )
                + 2 * u[t, x, y]
                - u[t - 1, x, y]
            )


fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set_xlim([0, 16])
    ax.set_ylim([0, 9])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Wave Animation")
    ax.contourf(x_n, y_n, u[i], cmap="inferno")
    return ax


anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=50)
anim.save("2D_wave_animation.mp4", writer="ffmpeg", fps=30)

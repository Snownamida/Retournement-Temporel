import numpy as np
from numpy import sin, cos, pi
from scipy.signal import fftconvolve
import time

from paramètres import *

u = np.zeros([Nt, Nx, Ny])
print(f"etimated size: {u.nbytes/1024**2:.2f} MB")


Lap_kernel = 0.5 * np.array(
    [
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5],
    ]
)


def laplacian(u_t):
    return fftconvolve(u_t, Lap_kernel, mode="same") / dl**2


u[0, Nx // 2, Ny // 2] = np.sin(0)
u[1, Nx // 2, Ny // 2] = np.sin(1 / 10)

print("Emulating...")
t0 = time.time()
for n in range(1, Nt):
    if not n % 10:
        t1 = time.time()
        print(
            f"\r{n}/{Nt} le temps reste estimé : {(Nt-n)*(t1-t0)/10:.2f} s",
            end="",
            flush=True,
        )
        t0 = t1
    Lap_u = laplacian(u[n - 1])
    u[n] = dt**2 * c**2 * Lap_u + 2 * u[n - 1] - u[n - 2]

    if n * dt < 0.3:
        u[n, Nx // 2, Ny // 2] = np.sin(50 * n * dt)
print("\ndone")

print("Saving...")
np.savez_compressed("./wave/" + para_string, u=u)
print("done")

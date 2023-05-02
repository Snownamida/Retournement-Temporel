import numpy as np
from numpy import sin, cos, pi
from scipy.signal import fftconvolve
import time

from paramètres import *


u_extended = np.zeros([Nt, Nx + 2 * N_absorb, Ny + 2 * N_absorb])
u = u_extended[:, N_absorb:-N_absorb, N_absorb:-N_absorb]
α = np.zeros_like(u_extended[0])

α[0:N_absorb] += np.linspace(α_max, 0, N_absorb)[:, None]
α[-N_absorb:] += np.linspace(0, α_max, N_absorb)[:, None]
α[:, 0:N_absorb] += np.linspace(α_max, 0, N_absorb)
α[:, -N_absorb:] += np.linspace(0, α_max, N_absorb)

print(f"etimated size: {u.nbytes/1024**2:.2f} MB")


Lap_kernel = 0.5 * np.array(
    [
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5],
    ]
)


def laplacian(u_t):
    Lap_u = fftconvolve(u_t, Lap_kernel, mode="same") / dl**2
    return Lap_u


print("Emulating...")
t0 = time.time()
for n in range(Nt):
    if not n % 10:
        t1 = time.time()
        print(
            f"\r{n}/{Nt} le temps reste estimé : {(Nt-n)*(t1-t0)/10:.2f} s",
            end="",
            flush=True,
        )
        t0 = t1

    Lap_u = laplacian(u_extended[n - 1])
    if n >= 2:
        u_extended[n] = (
            dt**2 * (c**2 * Lap_u - α * (u_extended[n - 1] - u_extended[n - 2]))
            + 2 * u_extended[n - 1]
            - u_extended[n - 2]
        )

    if n * dt <= 2 * pi / 70:
        for i_source, j_source in source_indices:
            u[n, i_source, j_source] = np.sin(70 * n * dt)

    if n * dt >= 1.5:
        u[n] = np.where(coeur, u[2 * int(1.5 / dt) - n], u[n])

    # print(u[n, 56, 158])


print("\ndone")

print("Saving...")
np.savez_compressed("./wave/" + para_string, u=u)
print("done")

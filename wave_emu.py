import numpy as np
from numpy import sin, cos, pi

from paramètres import *



u = np.zeros([Nt, Nx, Ny])


def laplacian(u_t, i, j):
    res = 0
    for i_adj, j_adj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
        if 0 <= i_adj < Nx and 0 <= j_adj < Ny:
            res += u_t[i_adj, j_adj] - u_t[i, j]
    return res / dl**2


u[0, Nx // 2, Ny // 2] = np.sin(0)
u[1, Nx // 2, Ny // 2] = np.sin(1 / 10)

for n in range(1, Nt):
    for i in range(Nx):
        for j in range(Ny):
            if n < 100:
                u[n, Nx // 2, Ny // 2] = np.sin(n / 10)
            u[n, i, j] = (
                dt**2 * c**2 * laplacian(u[n - 1], i, j)
                + 2 * u[n - 1, i, j]
                - u[n - 2, i, j]
            )

np.save("./wave/"+para_string, u)


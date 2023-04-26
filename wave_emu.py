import numpy as np
from numpy import sin, cos, pi

from paramètres import *

u = np.zeros([Nt, Nx, Ny])


def laplacian(u_t, i, j):
    res = 0
    if i not in (0,Nx-1) and j not in (0,Ny-1):
        for i_adj, j_adj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            res += u_t[i_adj, j_adj] - u_t[i, j]
        return res / dl**2
    elif (i,j)==(0,0):
        return laplacian(u_t,1,1)
    elif (i,j)==(0,Ny-1):
        return laplacian(u_t,1,Ny-2)
    elif (i,j)==(Nx-1,0):
        return laplacian(u_t,Nx-2,1)
    elif (i,j)==(Nx-1,Ny-1):
        return laplacian(u_t,Nx-2,Ny-2)
    elif i==0:
        return laplacian(u_t,1,j)
    elif i==Nx-1:
        return laplacian(u_t,Nx-2,j)
    elif j==0:
        return laplacian(u_t,i,1)
    elif j==Ny-1:
        return laplacian(u_t,i,Ny-2)
    else:
        raise ValueError("i or j is illegal")



u[0, Nx // 2, Ny // 2] = np.sin(0)
u[1, Nx // 2, Ny // 2] = np.sin(1 / 10)

for n in range(1, Nt):
    print(f"{n}/{Nt}",flush=True)
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


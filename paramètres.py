import numpy as np
from decimal import Decimal

c = 1.5  # wave speed

T = 0.5  # en secondes
Nt = 501
dt = T / (Nt - 1)

N_point = 101  # nombre de points minimum selon x ou y
Lx, Ly = 1.5, 1  # en mètres

dl = min(Lx, Ly) / (N_point - 1)

Nx, Ny = (int(Decimal(str(L)) // Decimal(str(dl))) + 1 for L in (Lx, Ly))
Lx, Ly = (Nx - 1) * dl, (Ny - 1) * dl
X, Y =np.meshgrid( np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

para_string=f"c={c}, T={T}, Nt={Nt}, N_point={N_point}, Lx={Lx}, Ly={Ly}"
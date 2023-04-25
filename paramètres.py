import numpy as np
Lx, Ly = 11,7 #en mètres
Nx, Ny = 12,8
X, Y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)
dx, dy = X[1] - X[0], Y[1] - Y[0]
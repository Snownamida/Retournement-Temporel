import numpy as np
from decimal import Decimal

# Paramètres de l'espace
Lx, Ly = 4, 3  # Largeur, longueur (m)
N_point = 201  # Nombre de points minimum selon x ou y


# Discrétisation
dl = min(Lx, Ly) / (
    N_point - 1
)  # Distance `dl` entre chaque point de l'espace. -1 car le (0;0) est pris en compte dans `N_point`
Nx, Ny = [int(L / dl) + 1 for L in (Lx, Ly)]
Lx, Ly = (Nx - 1) * dl, (
    Ny - 1
) * dl  # Recalcul des longueurs de effectives l'espace à partir des nouveaux nombres de points
X, Y = [grid.T for grid in np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))]

# Paramètres de simulation
c = 1.5  # Vitesse de propagation des ondes dans le milieu (m/s)
T = 4  # Temps final de simulation (s)
Nt = 1001  # Nombre d'itérations
dt = T / (Nt - 1)  # Pas de temps (s)
α = 1000  # Coefficient d'amortissement
L_absorb = 1
N_absorb = int(L_absorb / dl)  # Nombre de points absorbants aux bords

# Chaîne de caractères pour le nom du fichier
para_string = f"c={c}, T={T}, Nt={Nt}, N_point={N_point}, Lx={Lx}, Ly={Ly}, α={α}, n_absorb={N_absorb}"

#para de capteur
width = 0.001
a, b = 2, 1.5
coeur_size = 0.8
coeur_fun = ((X - a) / 1.3) ** 2 + ((Y - b) - (np.abs(X - a) / 1.3) ** (2 / 3)) ** 2
coeur = (coeur_fun <= coeur_size + width) & (coeur_fun >= coeur_size - width)
# print(np.sum(coeur))
# print(coeur[56,158])


#para de source
x_source,y_source=1.8,2
i_source,j_source=int(x_source/dl),int(y_source/dl)
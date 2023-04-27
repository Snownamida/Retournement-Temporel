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


# Paramètres de simulation
c = 1.5  # Vitesse de propagation des ondes dans le milieu (m/s)
T = 2  # Temps final de simulation (s)
Nt = 401  # Nombre d'itérations
dt = T / (Nt - 1)  # Pas de temps (s)
α = 1600  # Coefficient d'amortissement
L_absorb = 0.5
N_absorb = int(L_absorb / dl)  # Nombre de points absorbants aux bords

# Chaîne de caractères pour le nom du fichier
para_string = f"c={c}, T={T}, Nt={Nt}, N_point={N_point}, Lx={Lx}, Ly={Ly}, α={α}, n_absorb={N_absorb}"

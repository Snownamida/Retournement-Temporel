import numpy as np
from decimal import Decimal

# Paramètres de l'espace
Lx, Ly = 1.5, 1     # Largeur, longueur (m)
N_point = 101       # Nombre de points minimum selon x ou y


# Discrétisation
dl = min(Lx, Ly) / (N_point - 1)       # Distance `dl` entre chaque point de l'espace. -1 car le (0;0) est pris en compte dans `N_point`
Nx, Ny = [int(L/dl) + 1 for L in (Lx, Ly)]
Lx, Ly = (Nx - 1) * dl, (Ny - 1) * dl       # Recalcul des longueurs de effectives l'espace à partir des nouveaux nombres de points
X, Y = np.meshgrid( np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))     # Création de l'espace discrétisé



# Paramètres de simulation
c = 1.5             # Vitesse de propagation des ondes dans le milieu (m/s) 
T = 0.5             # Temps final de simulation (s)
Nt = 2001            # Nombre d'itérations
dt = T / (Nt - 1)   # Pas de temps (s)


# Vérifications
para_string=f"c={c}, T={T}, Nt={Nt}, N_point={N_point}, Lx={Lx}, Ly={Ly}"
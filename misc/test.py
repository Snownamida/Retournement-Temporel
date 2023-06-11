import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation

# Dimensions de l'espace
Lx = 15 # largeur
Ly = 15 # longueur


# Discrétisation
Nx = 100 # nombre de points sur l'axe X
Ny = 100 # nombre de points sur l'axe Y


# Construction des axes
X = np.linspace(0, Lx, Nx)   # Vecteur d'abscisses x
dx = X[2] - X[1]                # dx est la différence entre deux points sur x

Y = np.linspace(0, Ly, Ny)   # Vecteur d'ordonnées Y
dy = Y[2] - Y[1]                # dy est la différence entre deux points sur y

# Paramètres de simulation
Ti = 0                      # Temps initial de la simulation (s)
Tf = 15                     # Temps final de simulation (s)
Niter = 600                # Nombre d'itérations
dt = (Tf-Ti)/Niter          # Pas de temps (s)
v = 1                       # Vitesse de propagation des ondes dans le milieu (m/s) 


# Création du vecteur U
u = np.zeros([Niter, len(X), len(Y)])    # Stockage des différentes valeurs de u. Pour chaque itération, un nouveau vecteur de taille (X, Y) décrivant l'espace est créé. u est le vecteur de ces vecteurs (il y en a Niter au total).


###############################
### PARAMETRAGE DES SOURCES ###
###############################

# Création du vecteur de points exclus (pour conserver l'intégrité du signal source lors de la résolution)
excl = [(Nx // 2, Ny // 2)]         # Ici la source est une seule source ponctuelle située au centre

# Conditions initiales
u[0, Nx // 2, Ny // 2] = np.sin(0)       # Perturbation à t=0 au centre de l'espace
# u[1, Nx // 2, Ny // 2] = np.sin(1/10)  # Perturbation à t=1 au centre de l'espace

# Effets de la source
i = 1
while i < 100:
    u[i, Nx // 2, Ny // 2] = np.sin(i/10)
    i += 1


# t = i*dt
# while t < 3:
#     u[i, Nx // 2, Ny // 2] = np.sin(t*3)
#     i += 1
#     t=i*dt

# for i in range(0,Niter-1):      # On connaît déjà l'état de l'espace à t=0
#     print(f"Calcul pour i = {i+1}/{Niter-1}")
#     # Itération sur les X
#     for pos_x in range(1,Nx):
#         # Itération sur les Y
#         if (pos_x != 0) and (pos_x != len(X)-1):          # Si on n'est pas au bord X
#             for pos_y in range(1,Ny):
#                 if (((pos_y != 0) and (pos_y != len(Y)-1))) and ((pos_x, pos_y) not in excl):   # Si on n'est pas au bord Y
#                     u[i+1, pos_x, pos_y] = (dt*v)**2 * ( (u[i, pos_x+1, pos_y] - 2*u[i, pos_x, pos_y] + u[i, pos_x-1, pos_y])/(dx**2) + (u[i, pos_x, pos_y+1] - 2*u[i, pos_x, pos_y] + u[i, pos_x, pos_y-1])/(dy**2) ) + 2*u[i, pos_x, pos_y] - u[i-1, pos_x, pos_y]


for i in range(0,Niter-1):      # On connaît déjà l'état de l'espace à t=0, donc le calcul commencera à i+1 = 1
    print(f"Calcul pour i = {i+1}/{Niter}")
    if i != 200:
        v=1
    else:
        v=5
    for pos_x in range(1,Nx):   # Itération sur les X
        if (pos_x != 0) and (pos_x != len(X)-1):    # Si on n'est pas au bord X
            for pos_y in range(1,Ny):   # Itération sur les Y
                if (((pos_y != 0) and (pos_y != len(Y)-1))) and (((pos_x, pos_y) not in excl) or i>100):   # Si on n'est pas au bord Y ou que le point n'est pas exclu
                    # if i != 200:
                    u[i+1, pos_x, pos_y] = (dt*v)**2 * ( (u[i, pos_x+1, pos_y] - 2*u[i, pos_x, pos_y] + u[i, pos_x-1, pos_y])/(dx**2) + (u[i, pos_x, pos_y+1] - 2*u[i, pos_x, pos_y] + u[i, pos_x, pos_y-1])/(dy**2) ) + 2*u[i, pos_x, pos_y] - u[i-1, pos_x, pos_y]
                    # else:
                    #     u[i+1, pos_x, pos_y] = (dt*5)**2 * ( (u[i, pos_x+1, pos_y] - 2*u[i, pos_x, pos_y] + u[i, pos_x-1, pos_y])/(dx**2) + (u[i, pos_x, pos_y+1] - 2*u[i, pos_x, pos_y] + u[i, pos_x, pos_y-1])/(dy**2) ) + 2*u[i, pos_x, pos_y] - u[i-1, pos_x, pos_y]



fig = plt.figure()
minimum = u.min()
print(minimum)
maximum = u.max()
print(maximum)
titre = plt.title("Tracé pour t = 0s")
image = plt.imshow(u[0], cmap='magma', interpolation='bilinear', vmin=minimum, vmax=maximum)
interval = 4*((Tf-Ti)/Niter)
print(f"interval : {interval}")
plt.axis('off')

def updatefig(i):
    image.set_array(u[i])
    titre.set_text(f"Tracé pour t = {i*dt:.2f}s")
    return image, titre

ani = animation.FuncAnimation(fig, updatefig, frames=range(0, Niter), interval=interval, blit=False)
plt.show()



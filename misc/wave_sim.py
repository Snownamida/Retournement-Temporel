import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation



# Dimensions de l'espace
Lx = 16 # largeur
Ly = 9 # longueur


# Discrétisation
Nx = 80 # nombre de points sur l'axe X
Ny = 80 # nombre de points sur l'axe Y


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
u = np.zeros([Niter, len(Y), len(X)])    # Stockage des différentes valeurs de u. Pour chaque itération, un nouveau vecteur de taille (X, Y) décrivant l'espace est créé. u est le vecteur de ces vecteurs (il y en a Niter au total).


###############################
### PARAMETRAGE DES SOURCES ###
###############################

# Création du vecteur de points exclus (pour conserver l'intégrité du signal source lors de la résolution). Les points exclus sont ceux qui n'ont pas pour valeur `None`
init = np.full([Niter, len(Y), len(X)], None)

# Conditions initiales
u[0, Ny // 2, Nx // 2] = np.sin(0)       # Ici la source est une seule source ponctuelle située au centre. On initialise directement sur `u` car `init` décrit seulement la géometrie des soruces, et non pas l'espace
# u[0, 36, 59] = np.sin(np.pi/3)

# Effets de la source
i = 1
while i < 100:
    init[i, Ny // 2, Nx // 2] = np.sin(i/10)
    # init[i, Ny // 2, Nx // 2] = np.sin(i/10 + np.pi/3)
    i += 1



# WIP
def damping(espace : np.array, cptr : int, Nx : np.linspace, Ny : np.linspace):
    # Define damping factor
    damping_width = 10 # Width of sponge layer in grid points
    damping_strength = 0.9 # Strength of damping (0 = no damping, 1 = full damping)
    damping_x = np.ones(Nx)
    print(damping_x)
    damping_x[:damping_width] = np.linspace(1-damping_strength, 1, damping_width)
    damping_x[-damping_width:] = np.linspace(1, 1-damping_strength, damping_width)
    damping_y = np.ones(Ny)
    damping_y[:damping_width] = np.linspace(1-damping_strength, 1, damping_width)
    damping_y[-damping_width:] = np.linspace(1, 1-damping_strength, damping_width)

    # Apply sponge layer
    espace[cptr+1] *= damping_x[np.newaxis, :]
    espace[cptr+1] *= damping_y[:, np.newaxis]

#######################################
### RESOLUTION DE L'EQUATION D'ONDE ###
#######################################

# Résolution de l'EDP --> Probablement lente, on pourra voir à la fin pour optimiser
for i in range(0, Niter-1):         # On connaît déjà l'état de l'espace à t=0, donc le calcul commencera à `i+1 = 1`

    if not i%10: print(f"Calcul pour i = {i+1}/{Niter-1}")
    # abs_queue = []

    #### RETOURNEMENT ####
    if i != 200:
        v = 1
    else:
        v = 5

    # Itération sur les Y
    for pos_y in range(0,Ny):


        # Au bord haut en Y ? -> Dérivée partielle par rapport à Y avancée (vers le bas)
        if pos_y == 0:

            # Itération sur les X
            for pos_x in range(0,Nx):
                
                # # Ajout éventuel du point à la liste des points absorbants
                # if init[i+1, pos_y, pos_x] == "abs":
                #     abs_queue.append((pos_y, pos_x))

                # Au bord gauche en X ? -> Dérivée partielle par rapport à X avancée (à droite)
                if pos_x == 0:          
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y+1, pos_x] + 4*u[i, pos_y+2, pos_x] - u[i, pos_y+3, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x+1] + 4*u[i, pos_y, pos_x+2] - u[i, pos_y, pos_x+3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]  # Calcul si le point n'est pas exclu, égal à la condition initiale sinon.

                # Au bord droite en X ? -> Dérivée partielle par rapport à X retardée (à gauche)
                elif pos_x == len(X)-1:
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y+1, pos_x] + 4*u[i, pos_y+2, pos_x] - u[i, pos_y+3, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x-1] + 4*u[i, pos_y, pos_x-2] - u[i, pos_y, pos_x-3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]

                # Si on n'est pas au bord X
                else:    
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y+1, pos_x] + 4*u[i, pos_y+2, pos_x] - u[i, pos_y+3, pos_x])/(dy**2) + (u[i, pos_y, pos_x+1] - 2*u[i, pos_y, pos_x] + u[i, pos_y, pos_x-1])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]


        # Au bord bas en Y ? -> Dérivée partielle par rapport à Y retardée (vers le haut)
        elif pos_y == len(Y)-1:
                
                # Itération sur les X
                for pos_x in range(0,Nx):

                    # # Ajout éventuel du point à la liste des points absorbants
                    # if init[i+1, pos_y, pos_x] == "abs":
                    #     abs_queue.append((pos_y, pos_x))

                    # Au bord gauche en X ? -> Dérivée partielle par rapport à X avancée (à droite)
                    if pos_x == 0:          
                        u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y-1, pos_x] + 4*u[i, pos_y-2, pos_x] - u[i, pos_y-3, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x+1] + 4*u[i, pos_y, pos_x+2] - u[i, pos_y, pos_x+3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]

                    # Au bord droite en X ? -> Dérivée partielle par rapport à X retardée (à gauche)
                    elif pos_x == len(X)-1:
                        u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y-1, pos_x] + 4*u[i, pos_y-2, pos_x] - u[i, pos_y-3, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x-1] + 4*u[i, pos_y, pos_x-2] - u[i, pos_y, pos_x-3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]

                    # Si on n'est pas au bord X
                    else:    
                        u[i+1, pos_y, pos_x] = (dt*v)**2 * ((2*u[i, pos_y, pos_x] - 5*u[i, pos_y-1, pos_x] + 4*u[i, pos_y-2, pos_x] - u[i, pos_y-3, pos_x])/(dy**2) + (u[i, pos_y, pos_x+1] - 2*u[i, pos_y, pos_x] + u[i, pos_y, pos_x-1])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]


        # Si on n'est pas au bord Y
        else:

            # Itération sur les X
            for pos_x in range(0,Nx):

                # # Ajout éventuel du point à la liste des points absorbants
                # if init[i+1, pos_y, pos_x] == "abs":
                #     abs_queue.append((pos_y, pos_x))

                # Au bord gauche en X ? -> Dérivée partielle par rapport à X avancée (à droite)
                if pos_x == 0:    
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((u[i, pos_y+1, pos_x] - 2*u[i, pos_y, pos_x] + u[i, pos_y-1, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x+1] + 4*u[i, pos_y, pos_x+2] - u[i, pos_y, pos_x+3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]

                # Au bord droite en X ? -> Dérivée partielle par rapport à X retardée (à gauche)
                elif pos_x == len(X)-1:
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((u[i, pos_y+1, pos_x] - 2*u[i, pos_y, pos_x] + u[i, pos_y-1, pos_x])/(dy**2) + (2*u[i, pos_y, pos_x] - 5*u[i, pos_y, pos_x-1] + 4*u[i, pos_y, pos_x-2] - u[i, pos_y, pos_x-3])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]

                # Si on n'est pas au bord X
                else:    
                    u[i+1, pos_y, pos_x] = (dt*v)**2 * ((u[i, pos_y+1, pos_x] - 2*u[i, pos_y, pos_x] + u[i, pos_y-1, pos_x])/(dy**2) + (u[i, pos_y, pos_x+1] - 2*u[i, pos_y, pos_x] + u[i, pos_y, pos_x-1])/(dx**2)) + 2*u[i, pos_y, pos_x] - u[i-1, pos_y, pos_x] if ((init[i+1, pos_y, pos_x] == None) or (init[i+1, pos_y, pos_x] == "abs")) else init[i+1, pos_y, pos_x]
        # damping(u,i+1, Nx, Ny)


#################
### ANIMATION ###
#################

def animer(espace : np.array, dt : float, Niter : int, Ti : float, Tf : float,  interpol : str = "bilinear"):
    # Création de la figure et des paramètres
    fig = plt.figure()
    minimum = espace.min()       # Bornes pour la heatmap, sinon ça fait des flash
    print(minimum)
    maximum = espace.max()
    print(maximum)
    titre = plt.title("Tracé pour t = 0s")
    image = plt.imshow(espace[0], cmap='magma', interpolation=interpol, vmin=minimum, vmax=maximum)
    plt.axis('off')

    # Pour converser la vitesse de l'animation même si on change les paramètres de simulation.
    # En théorie ça marche, en pratique ça marche pas parce qu'on est limité par les performances de python et de la machine
    interval = 4*((Tf-Ti)/Niter)

    # Fonction de mise à jour de l'animation
    def updatefig(i):
        image.set_array(espace[i])
        titre.set_text(f"Tracé pour t = {i*dt:.2f}s")
        return image, titre

    # Animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(0, Niter), interval=interval, blit=False)
    plt.show()


animer(u, dt, Niter, Ti, Tf)
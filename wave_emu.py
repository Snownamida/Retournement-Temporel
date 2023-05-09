import numpy as np
from numpy import sin, cos, pi
from scipy.signal import fftconvolve
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def laplacian(u_t, dl):
    Lap_kernel = 0.5 * np.array(
        [
            [0.5, 1, 0.5],
            [1, -6, 1],
            [0.5, 1, 0.5],
        ]
    )
    Lap_u = fftconvolve(u_t, Lap_kernel, mode="same") / dl**2
    return Lap_u


class Onde:
    Lx, Ly = 4, 3  # Largeur, longueur (m)
    N_point = 401  # Nombre de points minimum selon x ou y
    c = 1.5  # Vitesse de propagation des ondes dans le milieu (m/s)
    T = 4  # Temps final de simulation (s)
    Nt = 1001  # Nombre d'itérations
    α_max = 20  # Coefficient d'amortissement
    L_absorb = 1

    def __init__(self, save_data, render_only) -> None:
        self.discretize()
        self.create_capteurs()
        self.create_sources()
        if not render_only:
            self.emulate()
        if save_data:
            self.save()
        self.render(render_only)

    def discretize(self):
        # Distance `dl` entre chaque point de l'espace. -1 car le (0;0) est pris en compte dans `N_point`
        self.dl = min(self.Lx, self.Ly) / (self.N_point - 1)
        self.Nx, self.Ny = [int(L / self.dl) + 1 for L in (self.Lx, self.Ly)]

        # Recalcul des longueurs de effectives l'espace à partir des nouveaux nombres de points
        self.Lx, self.Ly = (self.Nx - 1) * self.dl, (self.Ny - 1) * self.dl

        self.X, self.Y = [
            grid.T
            for grid in np.meshgrid(
                np.linspace(0, self.Lx, self.Nx), np.linspace(0, self.Ly, self.Ny)
            )
        ]

        self.dt = self.T / (self.Nt - 1)  # Pas de temps (s)
        # Nombre de points absorbants aux bords
        self.N_absorb = int(self.L_absorb / self.dl)

        # Chaîne de caractères pour le nom du fichier
        self.para_string = f"c={self.c}, T={self.T}, Nt={self.Nt}, N_point={self.N_point}, Lx={self.Lx}, Ly={self.Ly}, α={self.α_max}, n_absorb={self.N_absorb}"

    def create_capteurs(self):
        width = 0.001
        a, b = 2, 1.5
        coeur_size = 0.8
        coeur_fun = ((self.X - a) / 1.3) ** 2 + (
            (self.Y - b) - (np.abs(self.X - a) / 1.3) ** (2 / 3)
        ) ** 2
        self.coeur = (coeur_fun <= coeur_size + width) & (
            coeur_fun >= coeur_size - width
        )

    def create_sources(self):
        source_coordonnées = np.array(
            [
                [1, 1.2],
                [2, 2.2],
            ]
        )
        self.source_indices = np.rint(source_coordonnées / self.dl).astype(int)

    def emulate(self):
        u_extended = np.zeros(
            [self.Nt, self.Nx + 2 * self.N_absorb, self.Ny + 2 * self.N_absorb]
        )
        self.u = u_extended[
            :, self.N_absorb : -self.N_absorb, self.N_absorb : -self.N_absorb
        ]
        α = np.zeros_like(u_extended[0])

        α[0 : self.N_absorb] += np.linspace(self.α_max, 0, self.N_absorb)[:, None]
        α[-self.N_absorb :] += np.linspace(0, self.α_max, self.N_absorb)[:, None]
        α[:, 0 : self.N_absorb] += np.linspace(self.α_max, 0, self.N_absorb)
        α[:, -self.N_absorb :] += np.linspace(0, self.α_max, self.N_absorb)

        print(f"etimated size: {self.u.nbytes/1024**2:.2f} MB")

        print("Emulating...")
        t0 = time.time()
        for n in range(self.Nt):
            if not n % 10:
                t1 = time.time()
                print(
                    f"\r{n}/{self.Nt} le temps reste estimé : {(self.Nt-n)*(t1-t0)/10:.2f} s",
                    end="",
                    flush=True,
                )
                t0 = t1

            Lap_u = laplacian(u_extended[n - 1], self.dl)
            if n >= 2:
                u_extended[n] = (
                    self.dt**2
                    * (
                        self.c**2 * Lap_u
                        - α * (u_extended[n - 1] - u_extended[n - 2]) / self.dt
                    )
                    + 2 * u_extended[n - 1]
                    - u_extended[n - 2]
                )

            if n * self.dt <= 2 * pi / 70:
                for i_source, j_source in self.source_indices:
                    self.u[n, i_source, j_source] = sin(70 * n * self.dt)

            if n * self.dt >= 1.5:
                self.u[n] = np.where(
                    self.coeur, self.u[2 * int(1.5 / self.dt) - n], self.u[n]
                )

        print("\ndone")

    def save(self):
        print("Saving...")
        np.savez_compressed("./wave/" + self.para_string, u=self.u)
        print("done")

    def render(self, render_only) -> None:
        if render_only:
            u = np.load("./wave/" + self.para_string + ".npz")["u"]
        else:
            u = self.u

        fps = 40
        render_time = self.T  # temps de rendu

        # 1 seconde du temps réel correspond à combien seconde du temps de rendu
        render_speed = 0.3

        N_frame = int(fps * render_time / render_speed)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_xlim([0, self.Lx])
        ax.set_ylim([0, self.Ly])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        u_max = 0.1

        print("rendering...")
        self.t0 = time.time()

        u_img = ax.imshow(
            [[]],
            cmap="seismic",
            vmin=-u_max,
            vmax=u_max,
            extent=[0, self.Lx, 0, self.Ly],
            zorder=0,
        )
        coeur_img = ax.scatter([], [], c="r", s=1, zorder=5)

        def animate(n_frame):
            n = int(render_speed / self.dt / fps * n_frame)
            ax.set_title(f"t={n*self.dt:.5f}")
            u_img.set_data(u[n, :, ::-1].T)
            coeur_img.set_offsets(np.argwhere(self.coeur) * self.dl)
            if not n_frame % 10:
                t1 = time.time()
                print(
                    f"\r{n_frame}/{N_frame} le temps reste estimé : {(N_frame-n_frame)*(t1-self.t0)/10:.2f} s",
                    end="",
                    flush=True,
                )
                self.t0 = t1

            return u_img, coeur_img

        anim = animation.FuncAnimation(
            fig, animate, frames=N_frame, interval=50, blit=True
        )
        anim.save("./wave/" + self.para_string + ".mp4", writer="ffmpeg", fps=fps)
        print("\ndone")


onde = Onde(save_data=False, render_only=False)

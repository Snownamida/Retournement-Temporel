import numpy as np
from numpy import sin, cos, pi
from scipy.signal import fftconvolve
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def laplacian_con(u_t, dl):
    Lap_kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 16, 0, 0],
            [-1, 16, -60, 16, -1],
            [0, 0, 16, 0, 0],
            [0, 0, -1, 0, 0],
        ]
    ) / (12 * dl**2)
    Lap_u = fftconvolve(u_t, Lap_kernel, mode="same")
    return Lap_u


def laplacian_mat(u_t, dl):
    Nx, Ny = u_t.shape
    N = Nx * Ny
    diagonals = [
        -4 * np.ones(N),
        np.ones(N - 1),
        np.ones(N - 1),
        np.ones(N - Ny),
        np.ones(N - Ny),
    ]
    return sparse.diags(diagonals, [0, 1, -1, Ny, -Ny]) / dl**2


class Onde:
    Lx, Ly = 4, 3  # Largeur, longueur (m)
    N_point = 401  # Nombre de points minimum selon x ou y
    c = 1.5  # Vitesse de propagation des ondes dans le milieu (m/s)
    T = 0.03  # Temps final de simulation (s)
    Nt = 11  # Nombre d'itérations
    α_max = 20  # Coefficient d'amortissement
    L_absorb = 1
    T_emission = 2

    def __init__(self, save_data, render_only) -> None:
        self.discretize()
        self.create_capteurs()
        self.create_sources()
        self.create_simzone()
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

        self.n_emission = int(self.T_emission / self.dt)

        # Chaîne de caractères pour le nom du fichier
        self.para_string = f"c={self.c}, T={self.T}, Nt={self.Nt}, N_point={self.N_point}, Lx={self.Lx}, Ly={self.Ly}, α={self.α_max}, n_absorb={self.N_absorb}"

    def create_capteurs_coeur(self):
        width = 0.01
        a, b = 2, 1.5
        coeur_size = 0.8
        coeur_fun = ((self.X - a) / 1.3) ** 2 + (
            (self.Y - b) - (np.abs(self.X - a) / 1.3) ** (2 / 3)
        ) ** 2
        self.coeur = (coeur_fun <= coeur_size + width) & (
            coeur_fun >= coeur_size - width
        )

    def create_capteurs(self):
        width = 0.005
        a, b = 2, 1.5
        coeur_size = 1.4
        coeur_fun = ((self.X - a) ** 2 + (self.Y - b) ** 2) ** 0.5
        self.coeur = (coeur_fun <= coeur_size + width) & (
            coeur_fun >= coeur_size - width
        )

    def create_sources(self):
        source_coordonnées = np.array(
            [
                [1.9, 2.2],
                [2.5, 1],
            ]
        )
        self.source_indices = np.rint(source_coordonnées / self.dl).astype(int)

    def create_simzone(self):
        self.u = np.zeros(
            [self.Nt, self.Nx + 2 * self.N_absorb, self.Ny + 2 * self.N_absorb]
        )
        self.u_sim = self.u[
            :, self.N_absorb : -self.N_absorb, self.N_absorb : -self.N_absorb
        ]
        self.u_dot = np.zeros_like(self.u)
        self.α = np.pad(
            np.zeros_like(self.u_sim[0]),
            self.N_absorb,
            "linear_ramp",
            end_values=self.α_max,
        )
        self.Lap = laplacian_mat(self.u[0], self.dl)

    def udotdot(self, n):
        C = self.c**2 * self.Lap @ self.u[n]
        A = -self.α * (self.u[n] - self.u[n - 1]) / self.dt

        if self.n_emission <= n <= 2 * self.n_emission - 4:
            S = (
                -130
                * np.where(
                    self.coeur,
                    (
                        self.u_sim[2 * self.n_emission - n - 3]
                        - self.u_sim[2 * self.n_emission - n - 4]
                    ),
                    0,
                )
                / self.dt
            )

            S = np.pad(
                S,
                self.N_absorb,
                "constant",
                constant_values=0,
            )

        else:
            S = 0
        return C + A + S

    def fox_goodwin(self, n):
        Nx, Ny = self.u[0].shape
        N = Nx * Ny
        U = sparse.identity(N) - (self.dt * self.c) ** 2 / 12 * self.Lap
        V = (
            self.dt * self.u_dot[n - 1].flatten()
            + (sparse.identity(N) + 5 / 12 * (self.dt * self.c) ** 2 * self.Lap)
            @ self.u[n - 1].flatten()
        )
        self.u[n] = spsolve(
            U,
            V,
        ).reshape(Nx, Ny)
        self.u_dot[n] = self.u_dot[n - 1] + 0.5 * self.dt * self.c**2 * (
            self.Lap @ (self.u[n - 1] + self.u[n]).flatten()
        ).reshape(Nx, Ny)

    def emulate(self):
        print(f"etimated size: {self.u_sim.nbytes/1024**2:.2f} MB")

        print("Emulating...")
        t0 = time.time()
        for n in range(self.Nt):
            if not n % 20:
                t1 = time.time()
                print(
                    f"\r{n}/{self.Nt} le temps reste estimé : {(self.Nt-n)*(t1-t0)/20:.2f} s",
                    end="",
                    flush=True,
                )
                t0 = t1

            if n in (self.n_emission - 2, self.n_emission - 1):
                continue

            if n >= 1:
                self.fox_goodwin(n)

            T_source = 0.05
            if n * self.dt <= T_source:
                for i_source, j_source in self.source_indices:
                    self.u_sim[n, i_source, j_source] = 0.5 * sin(
                        pi / T_source * n * self.dt
                    )

        print("\ndone")

    def save(self):
        print("Saving...")
        np.savez_compressed("./wave/" + self.para_string, u=self.u_sim)
        print("done")

    def render(self, render_only) -> None:
        if render_only:
            u = np.load("./wave/" + self.para_string + ".npz")["u"]
        else:
            u = self.u_sim

        fps = 30
        render_time = self.T  # temps de rendu

        # 1 seconde du temps réel correspond à combien seconde du temps de rendu
        render_speed = 0.05

        N_frame = int(fps * render_time / render_speed)

        fig, ax = plt.subplots(figsize=(7, 5))
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

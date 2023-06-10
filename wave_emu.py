cuda = True  # True pour utiliser cupy, False pour utiliser

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate

if cuda:
    from cupy import (
        array,
        sin,
        cos,
        pi,
        ones,
        ones_like,
        meshgrid,
        linspace,
        abs,
        zeros,
        zeros_like,
        where,
        pad,
        arange,
        rint,
        load,
        argwhere,
    )
    from cupyx.scipy.signal import convolve
    from cupyx.scipy import sparse
    from cupyx.scipy.sparse.linalg import spsolve
    from cupyx.scipy.ndimage import laplace
else:
    from numpy import (
        array,
        sin,
        cos,
        pi,
        ones,
        ones_like,
        meshgrid,
        linspace,
        abs,
        zeros,
        zeros_like,
        where,
        pad,
        arange,
        rint,
        load,
        argwhere,
    )
    from scipy.signal import convolve
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import laplace


def cp_to_np(array):
    return array.get() if cuda else array


def laplacian_con(u_t, dl):
    Lap_kernel = (
        array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ]
        )
        / dl**2
    )
    Lap_u = convolve(u_t, Lap_kernel, mode="same")
    return Lap_u


def laplacian_sp(u_t, dl):
    return laplace(u_t) / dl**2


def laplacian_mat(u_t, dl):
    Nx, Ny = u_t.shape
    N = Nx * Ny
    diagonals = [
        -4 * ones(N),
        1 * ones(N - 1),
        1 * ones(N - 1),
        1 * ones(N - Ny),
        1 * ones(N - Ny),
    ]
    Lap = sparse.diags(diagonals, [0, 1, -1, Ny, -Ny]) / dl**2
    return (Lap @ u_t.flatten()).reshape(Nx, Ny)


class Onde:
    Lx, Ly = 4, 3  # Largeur, longueur (m)
    N_point = 541  # Nombre de points minimum selon x ou y
    c = 1  # Vitesse de propagation des ondes dans le milieu (m/s)
    T = 6  # Temps final de simulation (s)
    dt = 0.003
    α_max = 20  # Coefficient d'amortissement
    L_absorb = 1
    T_RT_duration = 3  # 只在读取外部数据有用,否则与T_RT_begins_at一致
    T_RT_begins_at = 3  # 只在自己模拟自己RT时有用,否则为0(一上来就RT)

    CcCcC = False  # True pour activer la variation de c

    n = 0  # Compteur d'itérations
    N_cache = 10  # on enrgistre l'onde pour combien de pas de temps

    读取外部数据 = False

    实时渲染 = False
    fps = 30
    # 1 seconde du temps réel correspond à combien seconde du temps de rendu
    # 只在保存视频时有用
    render_speed = 0.3

    def __init__(self) -> None:
        self.discretize()
        self.create_sources()
        self.create_simzone()
        self.create_capteurs()
        self.config_plot()
        self.render()

    def discretize(self):
        # Distance `dl` entre chaque point de l'espace. -1 car le (0;0) est pris en compte dans `N_point`
        self.dl = min(self.Lx, self.Ly) / (self.N_point - 1)
        self.Nx, self.Ny = [int(L / self.dl) + 1 for L in (self.Lx, self.Ly)]

        # Recalcul des longueurs de effectives l'espace à partir des nouveaux nombres de points
        self.Lx, self.Ly = (self.Nx - 1) * self.dl, (self.Ny - 1) * self.dl

        self.X, self.Y = [
            grid.T
            for grid in meshgrid(
                linspace(0, self.Lx, self.Nx), linspace(0, self.Ly, self.Ny)
            )
        ]
        self.Nt = int(self.T / self.dt) + 1
        self.T = (self.Nt - 1) * self.dt
        # Nombre de points absorbants aux bords
        self.N_absorb = int(self.L_absorb / self.dl)

        if self.读取外部数据:
            self.T_RT_begins_at = 0
        else:
            self.T_RT_duration = self.T_RT_begins_at
        self.N_RT = int(self.T_RT_duration / self.dt)
        self.n_RT_begins_at = int(self.T_RT_begins_at / self.dt)

        # Chaîne de caractères pour le nom du fichier
        self.para_string = f"c={self.c}, T={self.T}, Nt={self.Nt}, N_point={self.N_point}, Lx={self.Lx}, Ly={self.Ly}, α={self.α_max}, n_absorb={self.N_absorb}"

    def create_coeur(self, width=0.01, a=2, b=1.5, size=0.8):
        coeur_fun = ((self.X - a) / 1.3) ** 2 + (
            (self.Y - b) - (abs(self.X - a) / 1.3) ** (2 / 3)
        ) ** 2
        return (coeur_fun <= size + width) & (coeur_fun >= size - width)

    def create_cercle(self, width=0.005, a=2, b=1.5, size=1.4):
        cercle_fun = ((self.X - a) ** 2 + (self.Y - b) ** 2) ** 0.5
        return (cercle_fun <= size + width) & (cercle_fun >= size - width)

    def create_capteurs(self):
        if self.读取外部数据:
            mystère = load("mystère/mystère.npz")
            self.i_caps = mystère["i_caps"]
            self.j_caps = mystère["j_caps"]
            self.cap_données = mystère["cap_données"]
            self.cap_données = interpolate.interp1d(
                np.linspace(0, self.T_RT_duration, 256), cp_to_np(self.cap_données)
            )(np.linspace(0, self.T_RT_duration, self.N_RT))
            self.cap_forme = zeros_like(self.X, dtype=bool)
            self.cap_forme[self.i_caps, self.j_caps] = True
        else:
            self.cap_forme = self.create_cercle()
            self.i_caps, self.j_caps = where(self.cap_forme)
            self.cap_données = zeros((len(self.i_caps), self.N_RT))

    def u_cap(self, n):
        res = zeros_like(self.X)
        res[self.i_caps, self.j_caps] = self.cap_données[:, n]
        return res

    def create_sources(self):
        source_coordonnées = array(
            [
                [1.9, 2.2],
                # [2.5, 1],
            ]
        )
        self.source_indices = rint(source_coordonnées / self.dl).astype(int)
        self.i_sources, self.j_sources = self.source_indices.T

    def create_simzone(self):
        self.u = zeros(
            [self.N_cache, self.Nx + 2 * self.N_absorb, self.Ny + 2 * self.N_absorb]
        )
        self.u_sim = self.u[
            :, self.N_absorb : -self.N_absorb, self.N_absorb : -self.N_absorb
        ]
        self.u_dot = zeros_like(self.u)
        if self.CcCcC:
            self.c = (
                self.c * ones_like(self.u[0])
                + 0.2 * sin(2 * pi * arange(self.u.shape[1]) * self.dl / 2)[:, None]
            )
        self.α = pad(
            zeros_like(self.u_sim[0]),
            self.N_absorb,
            "linear_ramp",
            end_values=self.α_max,
        )

    def udotdot(self, n):
        C = self.c**2 * laplacian_sp(self.u[n % self.N_cache], self.dl)
        A = (
            -self.α
            * (self.u[n % self.N_cache] - self.u[n % self.N_cache - 1])
            / self.dt
        )

        if self.n_RT_begins_at <= n < self.n_RT_begins_at + self.N_RT:
            S = (
                -130
                * where(
                    self.cap_forme,
                    (
                        self.u_cap(-(n - self.n_RT_begins_at) - 1)
                        - self.u_cap(-(n - self.n_RT_begins_at))
                    ),
                    0,
                )
                / self.dt
            )

            S = pad(
                S,
                self.N_absorb,
                "constant",
                constant_values=0,
            )

        else:
            S = 0
        return C + A + S

    def config_plot(self):
        self.N_frame = int(self.fps * self.T / self.render_speed)

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(7, 5) if self.实时渲染 else (16, 9))
        self.ax.set_xlim([0, self.Lx])
        self.ax.set_ylim([0, self.Ly])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        u_max = 0.1

        self.u_img = self.ax.imshow(
            [[]],
            cmap="seismic",
            vmin=-u_max,
            vmax=u_max,
            extent=[0, self.Lx, 0, self.Ly],
            zorder=0,
            interpolation="none" if self.实时渲染 else "antialiased",
        )

        self.cap_img = self.ax.scatter([], [], c="r", s=1, zorder=5)

    def emulate(self, n_to_render):
        self.ax.set_title(f"t={n_to_render*self.dt:.5f}")

        while self.n <= n_to_render:
            if self.n == self.n_RT_begins_at and not self.读取外部数据:
                self.u[:] = 0

            if self.n >= 2:
                self.u[self.n % self.N_cache] = (
                    2 * self.u[(self.n - 1) % self.N_cache]
                    - self.u[(self.n - 2) % self.N_cache]
                    + self.dt**2 * self.udotdot(self.n - 1)
                )

            T_source = 0.05
            if self.n * self.dt < T_source and not self.读取外部数据:
                self.u_sim[
                    self.n % self.N_cache, self.i_sources, self.j_sources
                ] = 0.5 * sin(pi * self.n * self.dt / T_source)

            if self.n < self.N_RT and not self.读取外部数据:
                self.cap_données[:, self.n] = self.u_sim[
                    self.n % self.N_cache, self.i_caps, self.j_caps
                ]

            self.n += 1

        self.u_img.set_data(
            cp_to_np(self.u_sim[n_to_render % self.N_cache, ::1, ::-1].T)
        )

        self.cap_img.set_offsets(cp_to_np(argwhere(self.cap_forme) * self.dl))
        if not n_to_render % 10:
            t1 = time.time()
            print(
                f"\r{n_to_render}/{self.Nt} le temps reste estimé : {(self.Nt-n_to_render)*(t1-self.t0)/10:.2f} s",
                end="",
                flush=True,
            )
            self.t0 = t1
        return self.u_img, self.cap_img

    def render(self) -> None:
        print("emulating...")
        self.t0 = time.time()

        ns_to_render = [
            int(self.render_speed / self.dt / self.fps * n_frame)
            for n_frame in range(self.N_frame)
        ] + [self.Nt - 1]
        anim = animation.FuncAnimation(
            self.fig,
            self.emulate,
            frames=ns_to_render,
            interval=1,
            blit=True,
            repeat=False,
        )
        if self.实时渲染:
            plt.show()
        else:
            anim.save(
                "./wave/" + self.para_string + ".mp4", writer="ffmpeg", fps=self.fps
            )
        print("\ndone")


onde = Onde()

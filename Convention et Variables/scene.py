from manim import *

Lx, Ly = 11,7 #en mètres
Nx, Ny = 12,8
X, Y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)
dx, dy = X[1] - X[0], Y[1] - Y[0]

latex_dx = r"\mathop{\mathrm{d}x}"
latex_dy = r"\mathop{\mathrm{d}y}"
latex_dl = r"\mathop{\mathrm{d}l}"


class GridConfigure(Scene):
    def construct(self):
        # self.camera.background_color = WHITE

        self.scale_text = 0.6
        self.scale_text_title = 1.5
        self.runtime_text = 0.2

        self.draw_grid()
        self.draw_x_label()
        self.draw_y_label()
        self.draw_phrase()
        self.draw_formula()

    def draw_grid(self):
        self.grid = NumberPlane(
            x_range=(0, Lx + dx, dx),
            y_range=(0, Ly + dy, dy),
            # background_line_style={
            #     "stroke_color": BLUE_E,
            # "stroke_width": 4,
            #     "stroke_opacity": 1
            # }
        )
        self.grid.scale(0.7)
        self.grid.move_to(UP * 1)
        self.play(Create(self.grid))

        for i in range(Nx):
            for j in range(Ny):
                dot = Dot(
                    self.grid.coords_to_point(i * dx, j * dy), color=RED, radius=0.05
                )
                self.add(dot)

    def draw_x_label(self):
        x_pos = (
            np.array(
                [
                    0,
                    1,
                    2,
                    Nx // 2,
                    Nx - 2,
                    Nx - 1,
                ]
            )
            * dx
        )

        self.play(
            Write(
                MathTex("x", color=BLUE)
                .scale(self.scale_text * self.scale_text_title)
                .next_to(self.grid.coords_to_point(0, 0), LEFT + DOWN)
            ),
            run_time=self.runtime_text,
        )

        x_vals1 = [
            "0",
            latex_dx,
            "2" + latex_dx,
            r"\cdots",
            r"L_x\text{-}" + latex_dx,
            "L_x",
        ]
        for i in range(len(x_pos)):
            label = MathTex(x_vals1[i])
            label.scale(self.scale_text)
            label.next_to(self.grid.coords_to_point(x_pos[i], 0), DOWN)
            self.play(Write(label), run_time=self.runtime_text)

        self.play(
            Write(
                MathTex(r" \Rightarrow \text{distance}}")
                .scale(self.scale_text)
                .next_to(self.grid.coords_to_point(Lx + dx, 0), DOWN + RIGHT)
            ),
            run_time=self.runtime_text,
        )

        self.play(
            Write(
                MathTex("i", color=BLUE)
                .scale(self.scale_text * self.scale_text_title)
                .next_to(self.grid.coords_to_point(0, -0.9 * dx), LEFT + DOWN)
            ),
            run_time=self.runtime_text,
        )

        x_vals2 = [
            "0",
            "1",
            "2",
            r"\cdots",
            r"N_x\text{-}2",
            r"N_x\text{-}1",
        ]
        for i in range(len(x_pos)):
            label = MathTex(x_vals2[i])
            label.scale(self.scale_text)
            label.next_to(self.grid.coords_to_point(x_pos[i], -dx), DOWN)
            self.play(Write(label), run_time=self.runtime_text)

        self.play(
            Write(
                MathTex(r"\Rightarrow N_x\text{points au total}")
                .scale(self.scale_text)
                .next_to(self.grid.coords_to_point(Lx + dx, -dx), DOWN + RIGHT)
            ),
            run_time=self.runtime_text,
        )

    def draw_y_label(self):
        y_pos = (
            np.array(
                [
                    0,
                    1,
                    2,
                    Ny // 2,
                    Ny - 2,
                    Ny - 1,
                ]
            )
            * dy
        )

        self.play(
            Write(
                MathTex("y", color=BLUE)
                .scale(self.scale_text * self.scale_text_title)
                .next_to(self.grid.coords_to_point(0, Ly + dy), LEFT)
            ),
            run_time=self.runtime_text,
        )

        y_vals1 = [
            "0",
            latex_dy,
            "2" + latex_dy,
            r"\vdots",
            r"L_y\text{-}" + latex_dy,
            "L_y",
        ]
        for i in range(len(y_pos)):
            label = MathTex(y_vals1[i])
            label.scale(self.scale_text)
            label.next_to(self.grid.coords_to_point(0, y_pos[i]), LEFT)
            self.play(Write(label), run_time=self.runtime_text)

        self.play(
            Write(
                MathTex("j", color=BLUE)
                .scale(self.scale_text * self.scale_text_title)
                .next_to(self.grid.coords_to_point(-2 * dy, Ly + dy), LEFT)
            ),
            run_time=self.runtime_text,
        )

        y_vals2 = [
            "0",
            "1",
            "2",
            r"\vdots",
            r"N_y\text{-}2",
            r"N_y\text{-}1",
        ]
        for i in range(len(y_pos)):
            label = MathTex(y_vals2[i])
            label.scale(self.scale_text)
            label.next_to(self.grid.coords_to_point(-2 * dy, y_pos[i]), LEFT)
            self.play(Write(label), run_time=self.runtime_text)

    def draw_phrase(self):
        phrase1 = Tex(r"$x,y$ sont les coordonnées spatiales en mètres,", color=YELLOW)
        phrase2 = Tex(r"$i,j$ sont les indices selon x et y,", color=YELLOW)
        self.play(
            Write(phrase1.scale(self.scale_text).move_to(3.2 * DOWN)),
            Write(phrase2.scale(self.scale_text).next_to(phrase1, DOWN)),
            run_time=self.runtime_text,
        )

    def draw_formula(self):
        formulas = [
            MathTex(formula, color=YELLOW)
            for formula in [
                latex_dx + r"=\frac{L_x}{N_x-1}",
                latex_dy + r"=\frac{L_y}{N_y-1}",
                r"x=i\cdot" + latex_dx,
                r"y=j\cdot" + latex_dy,
                latex_dl+'='+latex_dx+'='+latex_dy
            ]
        ]
        self.play(
            Write(formulas[0].scale(self.scale_text).move_to(UP*3+RIGHT*5.5)),
            *[ Write(formulas[i].scale(self.scale_text).next_to(formulas[i-1], DOWN)) for i in range(1,len(formulas))],
            run_time=self.runtime_text,
        )

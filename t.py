"""
=============================================================
  SUPERCONDUCTING QUBIT — Full Manim Simulation (FIXED LAYOUT)
  3 Phases: Circuit → Superposition (Bloch) → Measurement
  
  Run command:
      manim -pqh t.py QuantumQubitSimulation
=============================================================
"""

from manim import *
import numpy as np

# ── COLOUR PALETTE ────────────────────────────────────────
CYAN_Q   = "#00E5FF"
GOLD_Q   = "#FFD600"
PURPLE_Q = "#7C3AED"
GREEN_Q  = "#00FF88"
RED_Q    = "#FF4444"
NAVY_Q   = "#050A1A"
MUTED_Q  = "#7A8BA8"
WHITE_Q  = "#F0F4FF"
PINK_Q   = "#FF6AC1"

config.background_color = "#050A1A"


# ══════════════════════════════════════════════════════════
#  SCENE 1 — TITLE CARD
# ══════════════════════════════════════════════════════════
class TitleScene(Scene):
    def construct(self):
        badge = Text("QUANTUM COMPUTING  ·  PHYSICAL SIMULATION",
                     font="Courier New", font_size=13, color=CYAN_Q,
                     weight=BOLD).to_edge(UP, buff=0.5)
        badge_box = SurroundingRectangle(badge, color=CYAN_Q,
                                          buff=0.12, stroke_width=1,
                                          corner_radius=0.12)

        title = VGroup(
            Text("Superconducting", font_size=52, weight=BOLD, color=WHITE_Q),
            Text("Qubit", font_size=80, weight=BOLD, color=CYAN_Q),
        ).arrange(DOWN, buff=0.1).shift(UP * 0.3)

        subtitle = Text(
            "From Josephson Junction  →  Superposition  →  Measurement",
            font_size=18, color=MUTED_Q, font="Courier New"
        ).next_to(title, DOWN, buff=0.4)

        phases = VGroup(*[
            VGroup(
                Text(f"Phase {i+1}", font_size=14, weight=BOLD,
                     color=CYAN_Q, font="Courier New"),
                Text(label, font_size=11, color=MUTED_Q, font="Courier New"),
            ).arrange(DOWN, buff=0.08)
            for i, label in enumerate([
                "Circuit & Cooper Pairs",
                "Bloch Sphere & Pulses",
                "Collapse & Histogram"
            ])
        ]).arrange(RIGHT, buff=0.6).next_to(subtitle, DOWN, buff=0.5)

        boxes = VGroup(*[
            SurroundingRectangle(p, color=CYAN_Q, buff=0.18,
                                  stroke_width=1, corner_radius=0.1,
                                  fill_color=NAVY_Q, fill_opacity=0.6)
            for p in phases
        ])

        ring = VGroup(*[
            Dot(radius=0.04, color=CYAN_Q, fill_opacity=0.6).move_to(
                2.8 * np.array([np.cos(a), np.sin(a), 0])
            )
            for a in np.linspace(0, TAU, 40, endpoint=False)
        ])

        self.play(Write(badge), Create(badge_box), run_time=1)
        self.play(FadeIn(ring, lag_ratio=0.02), run_time=1.2)
        self.play(DrawBorderThenFill(title[0]), DrawBorderThenFill(title[1]), run_time=1.8)
        self.play(FadeIn(subtitle, shift=UP*0.2), run_time=0.8)
        self.play(Create(boxes), FadeIn(phases, lag_ratio=0.3), run_time=1.2)
        self.play(ring.animate.rotate(PI/6), run_time=1.5)
        self.wait(1.5)
        self.play(FadeOut(VGroup(badge, badge_box, title, subtitle, phases, boxes, ring)))


# ══════════════════════════════════════════════════════════
#  SCENE 2 — PHASE 1: SUPERCONDUCTING CIRCUIT
# ══════════════════════════════════════════════════════════
class CircuitScene(Scene):
    def construct(self):
        self._phase_badge("01", "Qubit Representation — The Artificial Atom")

        circuit = self._build_circuit()
        self.play(Create(circuit["loop"]), run_time=1.5)
        self.play(Create(circuit["jj_group"]), run_time=1.2)
        self.play(Create(circuit["cap_group"]), run_time=1.0)
        self.play(FadeIn(circuit["labels"], lag_ratio=0.15), run_time=1.5)

        formula = self._formula_panel()
        self.play(FadeIn(formula, shift=RIGHT*0.3), run_time=1)

        self.play(FadeIn(circuit["jj_label"]), run_time=0.5)
        self._animate_cooper_pairs(circuit["jj_center"])

        energy_diag = self._energy_levels()
        self.play(FadeIn(energy_diag, shift=UP*0.3), run_time=1.2)
        self.wait(1)

        anharmonic = Text("Anharmonic Oscillator — |2⟩ NOT reachable at ω₀₁",
                          font_size=12, color=GOLD_Q,
                          font="Courier New").to_edge(DOWN, buff=0.3)
        box = SurroundingRectangle(anharmonic, color=GOLD_Q,
                                    buff=0.12, corner_radius=0.1,
                                    stroke_width=1.5)
        self.play(Write(anharmonic), Create(box), run_time=1)
        self.wait(2)
        self.play(FadeOut(VGroup(*self.mobjects)))

    def _phase_badge(self, num, text):
        badge = VGroup(
            Text(f"PHASE {num} / 03", font_size=11, color=CYAN_Q,
                 font="Courier New", weight=BOLD),
            Text(text, font_size=14, color=WHITE_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.05).to_corner(UL, buff=0.3)
        line = Line(badge.get_left(), badge.get_right() + RIGHT*0.2,
                    color=CYAN_Q, stroke_width=1).next_to(badge, DOWN, buff=0.06)
        self.play(FadeIn(badge), Create(line), run_time=0.6)

    def _build_circuit(self):
        loop = Rectangle(width=3.2, height=2.2, color=CYAN_Q, stroke_width=3).move_to(LEFT*1.8 + UP*0.2)
        loop.set_fill(CYAN_Q, opacity=0.03)

        jj_center = loop.get_top()
        sc1 = Rectangle(width=0.5, height=0.28, color=CYAN_Q, stroke_width=2,
                         fill_color=CYAN_Q, fill_opacity=0.2).move_to(jj_center + LEFT*0.35)
        sc2 = Rectangle(width=0.5, height=0.28, color=CYAN_Q, stroke_width=2,
                         fill_color=CYAN_Q, fill_opacity=0.2).move_to(jj_center + RIGHT*0.35)
        barrier = Rectangle(width=0.14, height=0.28, color=GOLD_Q, stroke_width=2,
                              fill_color=GOLD_Q, fill_opacity=0.5).move_to(jj_center)
        cross1 = Line(barrier.get_corner(UL), barrier.get_corner(DR), color=GOLD_Q, stroke_width=1.5)
        cross2 = Line(barrier.get_corner(UR), barrier.get_corner(DL), color=GOLD_Q, stroke_width=1.5)

        jj_group = VGroup(sc1, sc2, barrier, cross1, cross2)
        jj_label = VGroup(
            Text("Josephson Junction", font_size=12, color=GOLD_Q, font="Courier New", weight=BOLD),
            Text("AlOₓ ~1 nm barrier", font_size=10, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.03).next_to(jj_center, UP, buff=0.25)

        cap_center = loop.get_bottom()
        plate1 = Line(cap_center + LEFT*0.35, cap_center + RIGHT*0.35, color=CYAN_Q, stroke_width=3).move_to(cap_center + DOWN*0.08)
        plate2 = Line(cap_center + LEFT*0.35, cap_center + RIGHT*0.35, color=CYAN_Q, stroke_width=3).move_to(cap_center + UP*0.08)
        cap_label = Text("C (Shunt)", font_size=10, color=MUTED_Q, font="Courier New").next_to(cap_center, RIGHT, buff=0.4)
        cap_group = VGroup(plate1, plate2, cap_label)

        labels = VGroup(
            Text("Superconductor Al", font_size=10, color=CYAN_Q, font="Courier New").next_to(loop.get_left(), LEFT, buff=0.12),
            Text("T < 1.2 K", font_size=9, color=MUTED_Q, font="Courier New").next_to(loop.get_left() + DOWN*0.25, LEFT, buff=0.12),
        )

        return {"loop": loop, "jj_group": jj_group, "cap_group": cap_group, "labels": labels, "jj_center": jj_center, "jj_label": jj_label}

    def _formula_panel(self):
        formulas = VGroup(
            Text("Josephson Energy:", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"E_J = \frac{\Phi_0}{2\pi} I_c", color=CYAN_Q, font_size=24),
            Text("Charging Energy:", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"E_C = \frac{e^2}{2C}", color=CYAN_Q, font_size=24),
            Text("Qubit frequency:", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"\omega_{01} \approx \frac{\sqrt{8E_J E_C}}{\hbar} - \frac{E_C}{\hbar}", color=GREEN_Q, font_size=20),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        panel_box = SurroundingRectangle(formulas, color=PURPLE_Q, buff=0.2, corner_radius=0.12, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.8)
        return VGroup(panel_box, formulas).to_edge(RIGHT, buff=0.4).shift(UP * 0.2)

    def _animate_cooper_pairs(self, jj_center):
        title = Text("Cooper Pair Tunneling (2e⁻)", font_size=12, color=CYAN_Q, font="Courier New").to_edge(DOWN, buff=1.2)
        self.play(FadeIn(title))

        for _ in range(3):
            pair = VGroup(
                Dot(radius=0.08, color=CYAN_Q, fill_opacity=0.9).move_to(jj_center + LEFT*0.9),
                Dot(radius=0.08, color=CYAN_Q, fill_opacity=0.9).move_to(jj_center + LEFT*0.7),
            )
            bond = always_redraw(lambda p=pair: Line(p[0].get_right(), p[1].get_left(), color=CYAN_Q, stroke_width=1.5))
            self.add(pair, bond)

            self.play(pair.animate.shift(RIGHT * 1.6), run_time=0.75, rate_func=linear)
            self.remove(pair, bond)

        self.play(FadeOut(title))

    def _energy_levels(self):
        axes = Axes(
            x_range=[-2.0, 2.0, 1], y_range=[0, 4.5, 1],
            x_length=2.2, y_length=2.2,
            axis_config={"color": MUTED_Q, "stroke_width": 1.5, "include_ticks": False},
        ).to_corner(DL, buff=0.6).shift(UP*0.5)

        well = axes.plot(lambda x: 0.5*x**2 + 0.08*x**4, color=PURPLE_Q, stroke_width=2, x_range=[-1.9, 1.9])

        levels = [(0.28, "|0⟩", CYAN_Q), (1.45, "|1⟩", GOLD_Q), (3.20, "|2⟩", PURPLE_Q)]
        level_objs = VGroup()
        for e_val, lbl, col in levels:
            line = DashedLine(axes.c2p(-1.3, e_val), axes.c2p(1.3, e_val), color=col, stroke_width=1.5, dash_length=0.07)
            label = Text(lbl, font_size=9, color=col, font="Courier New").next_to(line, RIGHT, buff=0.08)
            level_objs.add(line, label)

        arr = Arrow(axes.c2p(1.0, 0.28), axes.c2p(1.0, 1.45), buff=0, color=CYAN_Q, stroke_width=1.5, tip_length=0.12)
        freq_lbl = MathTex(r"\hbar\omega_{01}", color=CYAN_Q, font_size=14).next_to(arr, RIGHT, buff=0.04)

        return VGroup(axes, well, level_objs, arr, freq_lbl)


# ══════════════════════════════════════════════════════════
#  SCENE 3 — PHASE 2: BLOCH SPHERE & PULSES
# ══════════════════════════════════════════════════════════
class BlochScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65*DEGREES, theta=-80*DEGREES)

        badge = Text("PHASE 02 / 03  —  Superposition via Microwave Pulses", font_size=12, color=CYAN_Q, font="Courier New", weight=BOLD).to_corner(UL, buff=0.35)
        self.add_fixed_in_frame_mobjects(badge)
        self.play(FadeIn(badge))

        sphere = self._build_sphere()
        self.play(*[Create(m) for m in sphere.values()], run_time=2.0)

        vector, dot_tip = self._make_vector(0)
        self.add(vector, dot_tip)

        lbl0 = MathTex(r"|0\rangle", color=CYAN_Q, font_size=26)
        lbl1 = MathTex(r"|1\rangle", color=GOLD_Q, font_size=26)
        self.add_fixed_in_frame_mobjects(lbl0, lbl1)
        lbl0.to_corner(UR, buff=0.5).shift(DOWN*0.8)
        lbl1.next_to(lbl0, DOWN, buff=0.3)

        state_text = Text("|0⟩ Ground State — Bloch North Pole", font_size=13, color=WHITE_Q, font="Courier New").to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(state_text)
        self.play(FadeIn(state_text))
        self.wait(1)

        self.play(
            Rotate(vector, angle=PI, axis=RIGHT, about_point=ORIGIN, rate_func=smooth),
            Rotate(dot_tip, angle=PI, axis=RIGHT, about_point=ORIGIN, rate_func=smooth),
            run_time=2.0
        )
        self.wait(1)

        self.play(
            Rotate(vector, angle=-PI/2, axis=RIGHT, about_point=ORIGIN, rate_func=smooth),
            Rotate(dot_tip, angle=-PI/2, axis=RIGHT, about_point=ORIGIN, rate_func=smooth),
            run_time=1.5
        )

        super_text = MathTex(r"|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle", color=GREEN_Q, font_size=22).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(super_text)
        self.play(Transform(state_text, super_text))

        self.wait(2)
        self.play(FadeOut(VGroup(*self.mobjects)))

    def _build_sphere(self):
        sphere_surf = Sphere(radius=1.0, resolution=(14, 14)).set_fill(NAVY_Q, opacity=0.08).set_stroke(CYAN_Q, width=0.4, opacity=0.25)
        z_axis = Arrow3D(DOWN*1.2, UP*1.3, color=CYAN_Q, thickness=0.02, base_radius=0.03)
        x_axis = Arrow3D(LEFT*1.2, RIGHT*1.2, color=MUTED_Q, thickness=0.01, base_radius=0.02)
        y_axis = Arrow3D(IN*1.2, OUT*1.2, color=MUTED_Q, thickness=0.01, base_radius=0.02)

        return {"sphere": sphere_surf, "z_ax": z_axis, "x_ax": x_axis, "y_ax": y_axis}

    def _make_vector(self, theta):
        tip = np.array([np.sin(theta), 0, np.cos(theta)])
        vec = Arrow3D(ORIGIN, tip, color=GREEN_Q, thickness=0.03, base_radius=0.04)
        dot = Sphere(radius=0.07, resolution=(8,8)).move_to(tip).set_fill(GREEN_Q, opacity=1)
        return vec, dot


# ══════════════════════════════════════════════════════════
#  SCENE 4 — PHASE 3: MEASUREMENT & HISTOGRAM
# ══════════════════════════════════════════════════════════
class MeasurementScene(Scene):
    def construct(self):
        self._phase_badge("03", "Quantum Measurement — Wavefunction Collapse")

        psi_eq = MathTex(r"|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle",
                        color=GREEN_Q, font_size=26).to_edge(UP, buff=0.55).shift(RIGHT*1.2)
        self.play(Write(psi_eq), run_time=1.0)

        axes = self._build_axes()
        self.play(Create(axes), run_time=1)

        np.random.seed(42)
        results = np.random.choice([0, 1], size=200, p=[0.5, 0.5])
        self._animate_shots(axes, results)

        self.wait(2)
        self.play(FadeOut(VGroup(*self.mobjects)))

    def _phase_badge(self, num, text):
        badge = VGroup(
            Text(f"PHASE {num} / 03", font_size=11, color=CYAN_Q, font="Courier New", weight=BOLD),
            Text(text, font_size=14, color=WHITE_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.05).to_corner(UL, buff=0.3)
        line = Line(badge.get_left(), badge.get_right() + RIGHT*0.2, color=CYAN_Q, stroke_width=1).next_to(badge, DOWN, buff=0.06)
        self.play(FadeIn(badge), Create(line), run_time=0.6)

    def _build_axes(self):
        axes = Axes(
            x_range=[0, 3, 1], y_range=[0, 1.05, 0.25],
            x_length=4.2, y_length=2.2,
            axis_config={"color": MUTED_Q, "stroke_width": 2, "include_ticks": False},
        ).shift(UP*0.1 + LEFT*1.2)

        for frac in [0.25, 0.5, 0.75, 1.0]:
            lbl = Text(f"{int(frac*100)}%", font_size=10, color=MUTED_Q, font="Courier New")
            lbl.next_to(axes.c2p(0, frac), LEFT, buff=0.1)
            self.add(lbl)

        for x, lbl, col in [(1, "|0⟩", CYAN_Q), (2, "|1⟩", GOLD_Q)]:
            t = Text(lbl, font_size=15, color=col, font="Courier New", weight=BOLD)
            t.next_to(axes.c2p(x, 0), DOWN, buff=0.12)
            self.add(t)

        target = DashedLine(axes.c2p(0, 0.5), axes.c2p(3, 0.5), color=GOLD_Q, stroke_width=1.5, dash_length=0.08)
        self.add(target)
        return axes

    def _animate_shots(self, axes, results):
        counts = [0, 0]
        shot_batches = [1]*10 + [5]*10 + [10]*14

        bar0 = Rectangle(width=0.6, height=0.001, fill_color=CYAN_Q, fill_opacity=0.85, stroke_color=CYAN_Q, stroke_width=1.5).move_to(axes.c2p(1, 0), aligned_edge=DOWN)
        bar1 = Rectangle(width=0.6, height=0.001, fill_color=GOLD_Q, fill_opacity=0.85, stroke_color=GOLD_Q, stroke_width=1.5).move_to(axes.c2p(2, 0), aligned_edge=DOWN)
        self.add(bar0, bar1)

        counter = Text("n = 0", font_size=18, color=WHITE_Q, font="Courier New", weight=BOLD).to_corner(DR, buff=0.5)
        p0_lbl = Text("P(|0⟩) = —", font_size=12, color=CYAN_Q, font="Courier New").next_to(counter, UP, buff=0.2)
        p1_lbl = Text("P(|1⟩) = —", font_size=12, color=GOLD_Q, font="Courier New").next_to(p0_lbl, UP, buff=0.1)
        self.add(counter, p0_lbl, p1_lbl)

        shot_idx = 0
        for batch in shot_batches:
            for _ in range(batch):
                if shot_idx >= len(results): break
                counts[results[shot_idx]] += 1
                shot_idx += 1

            total = sum(counts)
            p0, p1 = counts[0] / total, counts[1] / total

            h0 = max(axes.c2p(0, p0)[1] - axes.c2p(0, 0)[1], 0.002)
            h1 = max(axes.c2p(0, p1)[1] - axes.c2p(0, 0)[1], 0.002)

            new_bar0 = Rectangle(width=0.6, height=h0, fill_color=CYAN_Q, fill_opacity=0.85, stroke_color=CYAN_Q, stroke_width=1.5).move_to(axes.c2p(1, 0), aligned_edge=DOWN)
            new_bar1 = Rectangle(width=0.6, height=h1, fill_color=GOLD_Q, fill_opacity=0.85, stroke_color=GOLD_Q, stroke_width=1.5).move_to(axes.c2p(2, 0), aligned_edge=DOWN)

            new_counter = Text(f"n = {total}", font_size=18, color=WHITE_Q, font="Courier New", weight=BOLD).to_corner(DR, buff=0.5)
            new_p0 = Text(f"P(|0⟩) = {p0*100:.1f}%", font_size=12, color=CYAN_Q, font="Courier New").next_to(new_counter, UP, buff=0.2)
            new_p1 = Text(f"P(|1⟩) = {p1*100:.1f}%", font_size=12, color=GOLD_Q, font="Courier New").next_to(new_p0, UP, buff=0.1)

            self.play(
                Transform(bar0, new_bar0), Transform(bar1, new_bar1),
                Transform(counter, new_counter), Transform(p0_lbl, new_p0), Transform(p1_lbl, new_p1),
                run_time=0.12
            )

        conv = Text("✓ Converged to 50/50 — Born Rule Confirmed", font_size=13, color=GREEN_Q, font="Courier New", weight=BOLD).to_edge(DOWN, buff=0.3)
        box = SurroundingRectangle(conv, color=GREEN_Q, buff=0.12, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.9)
        self.play(Create(box), Write(conv), run_time=1.0)


# ══════════════════════════════════════════════════════════
#  SCENE 5 — OUTRO
# ══════════════════════════════════════════════════════════
class OutroScene(Scene):
    def construct(self):
        title = Text("Simulation Complete", font_size=42, weight=BOLD, color=CYAN_Q, font="Courier New")
        sub = VGroup(
            Text("Phase 01 · Josephson Junction · Cooper Pair Tunneling", font_size=12, color=MUTED_Q, font="Courier New"),
            Text("Phase 02 · Bloch Sphere · π and π/2 Microwave Pulses", font_size=12, color=MUTED_Q, font="Courier New"),
            Text("Phase 03 · Measurement Collapse · Born Rule · Law of Large Numbers", font_size=12, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.18).next_to(title, DOWN, buff=0.5)

        line = Line(LEFT*4.5, RIGHT*4.5, color=CYAN_Q, stroke_width=1).next_to(sub, DOWN, buff=0.4)
        credit = Text("Built with Manim · Quantum Computing Simulation", font_size=11, color=PURPLE_Q, font="Courier New").next_to(line, DOWN, buff=0.25)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, lag_ratio=0.25), run_time=1.2)
        self.play(Create(line), FadeIn(credit), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(VGroup(*self.mobjects)))


# ══════════════════════════════════════════════════════════
#  MASTER SCENE — chains all scenes into one MP4 (FULLY FIXED)
# ══════════════════════════════════════════════════════════
class QuantumQubitSimulation(Scene):
    def construct(self):
        self._title_card()
        self._phase1_circuit()
        self._phase2_bloch()
        self._phase3_measurement()
        self._outro()

    def _phase_badge(self, num, text):
        badge = VGroup(
            Text(f"PHASE {num} / 03", font_size=11, color=CYAN_Q, font="Courier New", weight=BOLD),
            Text(text, font_size=14, color=WHITE_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.04).to_corner(UL, buff=0.3)
        line = Line(badge.get_left(), badge.get_right() + RIGHT*0.3, color=CYAN_Q, stroke_width=1).next_to(badge, DOWN, buff=0.06)
        self.play(FadeIn(badge), Create(line), run_time=0.5)
        return VGroup(badge, line)

    def _clear(self):
        mobs = [m for m in self.mobjects if isinstance(m, VMobject)]
        if mobs:
            self.play(FadeOut(*mobs), run_time=0.5)
        for m in list(self.mobjects):
            self.remove(m)

    # ─────────────────────────────────────────────────────
    def _title_card(self):
        ring = VGroup(*[
            Dot(radius=0.04, color=CYAN_Q, fill_opacity=0.5).move_to(
                2.5*np.array([np.cos(a), np.sin(a), 0])
            ) for a in np.linspace(0, TAU, 36, endpoint=False)
        ])

        title = VGroup(
            Text("Superconducting Qubit", font_size=48, weight=BOLD, color=WHITE_Q),
            Text("Physical Quantum Simulation", font_size=24, color=CYAN_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.18).shift(UP*0.2)

        sub = Text(
            "Josephson Junction  →  Bloch Sphere  →  Born Rule Measurement",
            font_size=15, color=MUTED_Q, font="Courier New"
        ).next_to(title, DOWN, buff=0.45)

        phase_list = VGroup(*[
            VGroup(
                Text(f"0{i+1}", font_size=18, weight=BOLD, color=CYAN_Q, font="Courier New"),
                Text(lbl, font_size=11, color=MUTED_Q, font="Courier New"),
            ).arrange(DOWN, buff=0.04)
            for i, lbl in enumerate(["Circuit & Cooper Pairs", "Bloch Sphere & Pulses", "Collapse & Histogram"])
        ]).arrange(RIGHT, buff=0.8).next_to(sub, DOWN, buff=0.45)

        boxes = VGroup(*[
            SurroundingRectangle(p, color=CYAN_Q, buff=0.16, stroke_width=1, corner_radius=0.1, fill_color=NAVY_Q, fill_opacity=0.6)
            for p in phase_list
        ])

        self.play(FadeIn(ring, lag_ratio=0.02), run_time=0.8)
        self.play(Write(title[0]), run_time=1.0)
        self.play(FadeIn(title[1]), run_time=0.6)
        self.play(FadeIn(sub, shift=UP*0.2), run_time=0.7)
        self.play(Create(boxes), FadeIn(phase_list, lag_ratio=0.25), run_time=1)
        self.play(ring.animate.rotate(PI/8), run_time=1.2)
        self.wait(1.5)
        self._clear()

    # ─────────────────────────────────────────────────────
    def _phase1_circuit(self):
        badge = self._phase_badge("01", "Qubit Representation — Artificial Atom")

        loop = Rectangle(width=3.2, height=2.2, color=CYAN_Q, stroke_width=3).move_to(LEFT*1.8 + UP*0.1)
        loop.set_fill(CYAN_Q, opacity=0.02)
        self.play(Create(loop), run_time=1.0)

        jc = loop.get_top()
        sc1 = Rectangle(width=0.48, height=0.26, color=CYAN_Q, stroke_width=2, fill_color=CYAN_Q, fill_opacity=0.25).move_to(jc + LEFT*0.32)
        sc2 = Rectangle(width=0.48, height=0.26, color=CYAN_Q, stroke_width=2, fill_color=CYAN_Q, fill_opacity=0.25).move_to(jc + RIGHT*0.32)
        barrier = Rectangle(width=0.14, height=0.26, color=GOLD_Q, stroke_width=2, fill_color=GOLD_Q, fill_opacity=0.55).move_to(jc)
        cx1 = Line(barrier.get_corner(UL), barrier.get_corner(DR), color=GOLD_Q, stroke_width=1.5)
        cx2 = Line(barrier.get_corner(UR), barrier.get_corner(DL), color=GOLD_Q, stroke_width=1.5)

        jj_lbl = VGroup(
            Text("Josephson Junction", font_size=11, color=GOLD_Q, font="Courier New", weight=BOLD),
            Text("AlOₓ ~1 nm barrier", font_size=9, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.03).next_to(jc, UP, buff=0.2)

        self.play(Create(sc1), Create(sc2), Create(barrier), Create(cx1), Create(cx2), run_time=0.8)
        self.play(FadeIn(jj_lbl), run_time=0.5)

        bc = loop.get_bottom()
        p1 = Line(bc+LEFT*0.38, bc+RIGHT*0.38, color=CYAN_Q, stroke_width=3).move_to(bc+UP*0.1)
        p2 = Line(bc+LEFT*0.38, bc+RIGHT*0.38, color=CYAN_Q, stroke_width=3).move_to(bc+DOWN*0.1)
        cap_lbl = Text("Shunt Capacitor C", font_size=10, color=MUTED_Q, font="Courier New").next_to(bc, RIGHT, buff=0.4)
        self.play(Create(p1), Create(p2), FadeIn(cap_lbl), run_time=0.6)

        sc_lbl = VGroup(
            Text("Al Superconductor", font_size=10, color=CYAN_Q, font="Courier New"),
            Text("T < 1.2 K", font_size=9, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.04).next_to(loop.get_left(), LEFT, buff=0.12)
        self.play(FadeIn(sc_lbl), run_time=0.5)

        # FIXED: Drive arrow shifted right to avoid collision with jj_lbl
        drive = Arrow(loop.get_top() + RIGHT*0.9 + UP*1.0, loop.get_top() + RIGHT*0.9 + UP*0.25,
                      color=GOLD_Q, buff=0, stroke_width=2, tip_length=0.18)
        drive_lbl = Text("ω_drive ≈ 5.1 GHz", font_size=9, color=GOLD_Q, font="Courier New").next_to(drive, RIGHT, buff=0.1)
        self.play(GrowArrow(drive), FadeIn(drive_lbl), run_time=0.7)

        formulas = VGroup(
            Text("Josephson Energy:", font_size=10, color=MUTED_Q, font="Courier New"),
            MathTex(r"E_J = \frac{\Phi_0}{2\pi}I_c", color=CYAN_Q, font_size=22),
            Text("Charging Energy:", font_size=10, color=MUTED_Q, font="Courier New"),
            MathTex(r"E_C = \frac{e^2}{2C}", color=CYAN_Q, font_size=22),
            Text("Qubit frequency:", font_size=10, color=MUTED_Q, font="Courier New"),
            MathTex(r"\omega_{01} \approx \frac{\sqrt{8E_JE_C}}{\hbar}", color=GREEN_Q, font_size=19),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        fbox = SurroundingRectangle(formulas, color=PURPLE_Q, buff=0.2, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.8)
        fpanel = VGroup(fbox, formulas).to_edge(RIGHT, buff=0.35).shift(UP*0.2)
        self.play(FadeIn(fpanel), run_time=0.7)

        cp_title = Text("Cooper Pair Tunneling (2e⁻)", font_size=11, color=CYAN_Q, font="Courier New").to_edge(DOWN, buff=1.0)
        self.play(FadeIn(cp_title))

        for _ in range(3):
            pair = VGroup(
                Dot(radius=0.08, color=CYAN_Q).move_to(jc + LEFT*0.8),
                Dot(radius=0.08, color=CYAN_Q).move_to(jc + LEFT*0.6),
            )
            bond = Line(pair[0].get_right(), pair[1].get_left(), color=CYAN_Q, stroke_width=1.5)
            self.add(pair, bond)
            self.play(pair.animate.shift(RIGHT*1.4), bond.animate.shift(RIGHT*1.4), run_time=0.45, rate_func=linear)
            self.remove(pair, bond)

        self.play(FadeOut(cp_title))

        axes = Axes(x_range=[-1.8, 1.8], y_range=[0, 4.5], x_length=2.0, y_length=2.0,
                    axis_config={"color": MUTED_Q, "stroke_width": 1.5, "include_ticks": False}).to_corner(DL, buff=0.4).shift(UP*0.4)

        well = axes.plot(lambda x: 0.5*x**2 + 0.09*x**4, color=PURPLE_Q, stroke_width=2, x_range=[-1.8, 1.8])

        lvls = [(0.22, "|0⟩", CYAN_Q), (1.38, "|1⟩", GOLD_Q), (3.0, "|2⟩", PURPLE_Q)]
        lvl_grp = VGroup()
        for ev, lbl, col in lvls:
            ln = DashedLine(axes.c2p(-1.2, ev), axes.c2p(1.2, ev), color=col, stroke_width=1.5, dash_length=0.06)
            lt = Text(lbl, font_size=9, color=col, font="Courier New").next_to(ln, RIGHT, buff=0.06)
            lvl_grp.add(ln, lt)

        arr01 = Arrow(axes.c2p(0.9, 0.22), axes.c2p(0.9, 1.38), buff=0, color=CYAN_Q, stroke_width=1.5, tip_length=0.12)
        arr_lbl = MathTex(r"\hbar\omega_{01}", color=CYAN_Q, font_size=13).next_to(arr01, RIGHT, buff=0.04)

        self.play(Create(axes), Create(well), run_time=0.6)
        self.play(FadeIn(lvl_grp), GrowArrow(arr01), Write(arr_lbl), run_time=0.6)

        self.wait(2)
        self._clear()

    # ─────────────────────────────────────────────────────
    def _phase2_bloch(self):
        badge = self._phase_badge("02", "Superposition via Microwave Pulses")

        # FIXED: Reduced radius R=1.4 and center shift UP*0.3 to prevent bottom collision
        cx, cy = -2.2, 0.3
        center = np.array([cx, cy, 0])
        R = 1.4

        sphere_circle = Circle(radius=R, color=CYAN_Q, stroke_width=2, stroke_opacity=0.5, fill_color=NAVY_Q, fill_opacity=0.15).move_to(center)
        equator = DashedLine(center+LEFT*R, center+RIGHT*R, color=CYAN_Q, stroke_width=1.2, stroke_opacity=0.5, dash_length=0.08)
        z_ax = Line(center+DOWN*(R+0.2), center+UP*(R+0.2), color=CYAN_Q, stroke_width=1.5, stroke_opacity=0.6)

        lbl_0 = MathTex(r"|0\rangle", color=CYAN_Q, font_size=24).next_to(center+UP*R, UP, buff=0.1)
        lbl_1 = MathTex(r"|1\rangle", color=GOLD_Q, font_size=24).next_to(center+DOWN*R, DOWN, buff=0.1)
        lbl_p = MathTex(r"|+\rangle", color=GREEN_Q, font_size=18).next_to(center+RIGHT*R, RIGHT, buff=0.12)

        self.play(Create(sphere_circle), Create(equator), Create(z_ax), run_time=1.0)
        self.play(FadeIn(lbl_0), FadeIn(lbl_1), FadeIn(lbl_p), run_time=0.6)

        vec_tip = center + UP*R
        vector = Arrow(center, vec_tip, buff=0, color=GREEN_Q, stroke_width=3.5, tip_length=0.18)
        tip_dot = Dot(radius=0.09, color=GREEN_Q).move_to(vec_tip)

        state_lbl = VGroup(
            Text("Initial State:", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"|\psi\rangle = |0\rangle", color=CYAN_Q, font_size=22),
        ).arrange(DOWN, buff=0.06).to_edge(RIGHT, buff=0.5).shift(UP*1.2)

        self.play(GrowArrow(vector), FadeIn(tip_dot), FadeIn(state_lbl), run_time=0.8)
        self.wait(0.6)

        def make_pulse_panel(title, desc, color):
            return VGroup(
                Text(title, font_size=12, color=color, font="Courier New", weight=BOLD),
                *[Text(d, font_size=10, color=MUTED_Q, font="Courier New") for d in desc],
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)

        pi_panel = make_pulse_panel("π-Pulse (Full Flip)", ["Duration: 120 ns", "Rotates θ: 0 → π", "Result: |0⟩ → |1⟩", "Equiv. to X (NOT) gate"], GOLD_Q)
        pi_box = SurroundingRectangle(pi_panel, color=GOLD_Q, buff=0.18, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.85)
        pi_group = VGroup(pi_box, pi_panel).to_edge(RIGHT, buff=0.4).shift(UP*0.1)
        self.play(FadeOut(state_lbl), FadeIn(pi_group), run_time=0.5)

        new_tip = center + DOWN*R
        new_vec = Arrow(center, new_tip, buff=0, color=GOLD_Q, stroke_width=3.5, tip_length=0.18)
        new_dot = Dot(radius=0.09, color=GOLD_Q).move_to(new_tip)

        arc = Arc(radius=R, start_angle=PI/2, angle=-PI, color=GOLD_Q, stroke_width=1.2, stroke_opacity=0.5).move_to(center)
        self.play(Create(arc), run_time=0.3)
        self.play(Transform(vector, new_vec), Transform(tip_dot, new_dot), run_time=1.5, rate_func=smooth)

        pi_result = VGroup(
            Text("State flipped to", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"|1\rangle", color=GOLD_Q, font_size=26),
        ).arrange(RIGHT, buff=0.12).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(pi_result), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(pi_group), FadeOut(pi_result), FadeOut(arc))

        reset_vec = Arrow(center, center+UP*R, buff=0, color=GREEN_Q, stroke_width=3.5, tip_length=0.18)
        reset_dot = Dot(radius=0.09, color=GREEN_Q).move_to(center+UP*R)
        self.play(Transform(vector, reset_vec), Transform(tip_dot, reset_dot), run_time=0.8, rate_func=smooth)

        half_panel = make_pulse_panel("π/2-Pulse (Superposition)", ["Duration: 60 ns", "Rotates θ: 0 → π/2", "Lands on equator", "P(|0⟩) = 50%  P(|1⟩) = 50%"], GREEN_Q)
        half_box = SurroundingRectangle(half_panel, color=GREEN_Q, buff=0.18, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.85)
        half_group = VGroup(half_box, half_panel).to_edge(RIGHT, buff=0.4).shift(UP*0.1)
        self.play(FadeIn(half_group), run_time=0.5)

        super_tip = center + RIGHT*R
        super_vec = Arrow(center, super_tip, buff=0, color=GREEN_Q, stroke_width=3.5, tip_length=0.18)
        super_dot = Dot(radius=0.09, color=GREEN_Q).move_to(super_tip)

        arc2 = Arc(radius=R, start_angle=PI/2, angle=-PI/2, color=GREEN_Q, stroke_width=1.2, stroke_opacity=0.5).move_to(center)
        self.play(Create(arc2), run_time=0.3)
        self.play(Transform(vector, super_vec), Transform(tip_dot, super_dot), run_time=1.5, rate_func=smooth)

        # FIXED: Bottom superposition equation shifted safely below Bloch sphere
        super_eq = VGroup(
            MathTex(r"|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle", color=GREEN_Q, font_size=22),
            Text("P(|0⟩) = 50%    P(|1⟩) = 50%", font_size=12, color=WHITE_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.4)

        eq_box = SurroundingRectangle(super_eq, color=GREEN_Q, buff=0.15, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.9)
        self.play(Create(eq_box), Write(super_eq), run_time=1.0)
        self.wait(2)
        self._clear()

    # ─────────────────────────────────────────────────────
    def _phase3_measurement(self):
        badge = self._phase_badge("03", "Quantum Measurement — Collapse")

        # FIXED: Shifted psi equation slightly right to avoid badge collision
        psi = MathTex(r"|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle",
                      color=GREEN_Q, font_size=26).to_edge(UP, buff=0.55).shift(RIGHT*1.2)
        self.play(Write(psi), run_time=0.8)

        meas_arr = Arrow(LEFT*0.4, RIGHT*0.4, color=GOLD_Q, stroke_width=2.5, tip_length=0.18).next_to(psi, DOWN, buff=0.2)
        meas_lbl = VGroup(
            MathTex(r"\hat{M}", color=GOLD_Q, font_size=22),
            Text("Dispersive Readout", font_size=10, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.04).next_to(meas_arr, RIGHT, buff=0.15)
        self.play(GrowArrow(meas_arr), FadeIn(meas_lbl), run_time=0.6)

        # FIXED: Lifted axes UP*0.1 to leave plenty of vertical room at bottom
        axes = Axes(
            x_range=[0, 3, 1], y_range=[0, 1.05, 0.25],
            x_length=4.2, y_length=2.2,
            axis_config={"color": MUTED_Q, "stroke_width": 2, "include_ticks": False},
        ).shift(UP*0.1 + LEFT*1.2)

        self.play(Create(axes), run_time=0.6)

        for frac, txt in [(0.25,"25%"),(0.5,"50%"),(0.75,"75%"),(1.0,"100%")]:
            l = Text(txt, font_size=9, color=MUTED_Q, font="Courier New").next_to(axes.c2p(0,frac), LEFT, buff=0.08)
            self.add(l)
        for xi, txt, col in [(1,"|0⟩",CYAN_Q),(2,"|1⟩",GOLD_Q)]:
            t = Text(txt, font_size=15, color=col, font="Courier New", weight=BOLD).next_to(axes.c2p(xi,0), DOWN, buff=0.12)
            self.add(t)

        tgt = DashedLine(axes.c2p(0,0.5), axes.c2p(3,0.5), color=GOLD_Q, stroke_width=1.2, dash_length=0.08)
        tgt_l = Text("50%", font_size=9, color=GOLD_Q, font="Courier New").next_to(axes.c2p(3,0.5), RIGHT, buff=0.08)
        self.add(tgt, tgt_l)

        born = VGroup(
            Text("Born Rule:", font_size=11, color=MUTED_Q, font="Courier New"),
            MathTex(r"P(k)=|\langle k|\psi\rangle|^2", color=CYAN_Q, font_size=18),
        ).arrange(DOWN, buff=0.08).to_edge(RIGHT, buff=0.5).shift(UP*0.2)
        self.play(FadeIn(born), run_time=0.5)

        np.random.seed(2024)
        results = np.random.choice([0,1], size=200, p=[0.5,0.5])
        counts  = [0, 0]

        bar0 = Rectangle(width=0.55, height=0.001, fill_color=CYAN_Q, fill_opacity=0.9, stroke_color=CYAN_Q, stroke_width=1.5).move_to(axes.c2p(1,0), aligned_edge=DOWN)
        bar1 = Rectangle(width=0.55, height=0.001, fill_color=GOLD_Q, fill_opacity=0.9, stroke_color=GOLD_Q, stroke_width=1.5).move_to(axes.c2p(2,0), aligned_edge=DOWN)
        self.add(bar0, bar1)

        n_lbl  = Text("n = 0", font_size=18, color=WHITE_Q, font="Courier New", weight=BOLD).to_corner(DR, buff=0.5)
        p0_lbl = Text("P(|0⟩) = —", font_size=12, color=CYAN_Q, font="Courier New").next_to(n_lbl, UP, buff=0.2)
        p1_lbl = Text("P(|1⟩) = —", font_size=12, color=GOLD_Q, font="Courier New").next_to(p0_lbl, UP, buff=0.1)
        self.add(n_lbl, p0_lbl, p1_lbl)

        batches = [1]*10 + [5]*10 + [10]*14
        idx = 0
        for bsz in batches:
            for _ in range(bsz):
                if idx >= len(results): break
                counts[results[idx]] += 1
                idx += 1
            tot = sum(counts)
            p0, p1 = counts[0]/tot, counts[1]/tot

            h0 = max(axes.c2p(0,p0)[1]-axes.c2p(0,0)[1], 0.002)
            h1 = max(axes.c2p(0,p1)[1]-axes.c2p(0,0)[1], 0.002)

            nb0 = Rectangle(width=0.55, height=h0, fill_color=CYAN_Q, fill_opacity=0.9, stroke_color=CYAN_Q, stroke_width=1.5).move_to(axes.c2p(1,0), aligned_edge=DOWN)
            nb1 = Rectangle(width=0.55, height=h1, fill_color=GOLD_Q, fill_opacity=0.9, stroke_color=GOLD_Q, stroke_width=1.5).move_to(axes.c2p(2,0), aligned_edge=DOWN)
            nn  = Text(f"n = {tot}", font_size=18, color=WHITE_Q, font="Courier New", weight=BOLD).to_corner(DR, buff=0.5)
            np0 = Text(f"P(|0⟩) = {p0*100:.1f}%", font_size=12, color=CYAN_Q, font="Courier New").next_to(nn, UP, buff=0.2)
            np1 = Text(f"P(|1⟩) = {p1*100:.1f}%", font_size=12, color=GOLD_Q, font="Courier New").next_to(np0, UP, buff=0.1)

            self.play(
                Transform(bar0, nb0), Transform(bar1, nb1),
                Transform(n_lbl, nn), Transform(p0_lbl, np0), Transform(p1_lbl, np1),
                run_time=0.12
            )

        # FIXED: Bottom convergence banner cleanly aligned at lower margin
        diff = abs(counts[0]/sum(counts)-0.5)
        conv_col = GREEN_Q if diff < 0.06 else GOLD_Q
        conv_txt = Text(
            "✓ Converged → 50/50 | Born Rule Confirmed" if diff<0.06 else f"Deviation: {diff*100:.1f}%",
            font_size=13, color=conv_col, font="Courier New", weight=BOLD
        ).to_edge(DOWN, buff=0.3)
        conv_box = SurroundingRectangle(conv_txt, color=conv_col, buff=0.12, corner_radius=0.1, stroke_width=1.5, fill_color=NAVY_Q, fill_opacity=0.9)
        self.play(Create(conv_box), Write(conv_txt), run_time=0.8)

        self.wait(2)
        self._clear()

    # ─────────────────────────────────────────────────────
    def _outro(self):
        title = Text("Simulation Complete", font_size=42, weight=BOLD, color=CYAN_Q)
        subs = VGroup(
            Text("Phase 01 · Josephson Junction & Cooper Pair Tunneling", font_size=12, color=MUTED_Q, font="Courier New"),
            Text("Phase 02 · Bloch Sphere · π-Pulse · π/2-Pulse", font_size=12, color=MUTED_Q, font="Courier New"),
            Text("Phase 03 · Born Rule Collapse · Law of Large Numbers", font_size=12, color=MUTED_Q, font="Courier New"),
        ).arrange(DOWN, buff=0.18).next_to(title, DOWN, buff=0.45)

        line = Line(LEFT*4.5, RIGHT*4.5, color=CYAN_Q, stroke_width=1).next_to(subs, DOWN, buff=0.4)
        credit = Text("Built with Manim · Quantum Computing Simulation", font_size=11, color=PURPLE_Q, font="Courier New").next_to(line, DOWN, buff=0.25)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(subs, lag_ratio=0.25), run_time=1.0)
        self.play(Create(line), FadeIn(credit), run_time=0.7)
        self.wait(2.5)
"""
Simulador de Física I — v2.1
Versión anterior + Panel de solución de ecuaciones
Requiere: pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ═══════════════════════════════════════════════════════════════════
#  PALETA  (idéntica a v2)
# ═══════════════════════════════════════════════════════════════════
BG        = "#0f1117"
PANEL     = "#1a1d26"
PANEL2    = "#1f2335"
ACCENT    = "#00d4aa"
ACCENT2   = "#00b890"
ACCENT3   = "#7ee8d0"
ACCENT_RED= "#ff4d6d"
ACCENT_YEL= "#ffd166"
ACCENT_BLU= "#4ecdc4"
TEXT      = "#e8f4f0"
SUBTEXT   = "#7a9e97"
SUCCESS   = "#00d4aa"
ERROR     = "#ff4d6d"
WARN      = "#ffd166"
BORDER    = "#2a3340"
BG_ENTRY  = "#141720"
SIDEBAR   = "#13161e"
LINE1     = "#00d4aa"
LINE2     = "#ffd166"
LINE3     = "#ff6b6b"
FM        = "JetBrains Mono"

# ═══════════════════════════════════════════════════════════════════
#  FÍSICA
# ═══════════════════════════════════════════════════════════════════

def cinematica_mrua(v0, a, t_max, angulo=0):
    ang_rad = math.radians(angulo)
    v0x = v0 * math.cos(ang_rad)
    v0y = v0 * math.sin(ang_rad)
    g   = 9.8
    t   = np.linspace(0, t_max, 500)
    if angulo == 0:
        return {"t": t, "x": v0x*t + 0.5*a*t**2,
                "v": v0x + a*t, "a": np.full_like(t, a)}
    else:
        x  = v0x * t
        y  = v0y * t - 0.5*g*t**2
        vy = v0y - g*t
        v  = np.sqrt(v0x**2 + vy**2)
        idx = np.where(y < 0)[0]
        if len(idx):
            c = idx[0]
            x, y, t, v = x[:c], y[:c], t[:c], v[:c]
        return {"t": t, "x": x, "y": y, "v": v}


def segunda_ley(m, fx, ff, t_max):
    Fnet = fx - ff
    a    = Fnet / m
    t    = np.linspace(0, t_max, 500)
    return {"t": t, "v": a*t, "x": 0.5*a*t**2,
            "a": np.full_like(t, a), "ax": a, "Fx": fx, "Fnet": Fnet}


# ═══════════════════════════════════════════════════════════════════
#  SOLVERS  (devuelven res, err, pasos[])
# ═══════════════════════════════════════════════════════════════════

def resolver_mrua(v0=None, vf=None, a=None, t=None, x=None):
    if sum(v is not None for v in [v0, vf, a, t, x]) < 3:
        return None, "Necesitas al menos 3 valores.", []
    steps = []
    try:
        if v0 is not None and a is not None and t is not None \
                and vf is None and x is None:
            steps += ["Ecuación:  vf = v₀ + a·t",
                      f"  vf = {v0} + ({a})·({t})",
                      f"  vf = {v0+a*t:.2f} m/s",
                      "Ecuación:  x = v₀·t + ½·a·t²",
                      f"  x = {v0}·{t} + 0.5·{a}·{t}²",
                      f"  x = {v0*t+0.5*a*t**2:.2f} m"]
            vf = v0 + a*t
            x  = v0*t + 0.5*a*t**2

        elif v0 is not None and vf is not None and t is not None \
                and a is None and x is None:
            steps += ["Ecuación:  a = (vf − v₀) / t",
                      f"  a = ({vf} − {v0}) / {t}",
                      f"  a = {(vf-v0)/t:.2f} m/s²",
                      "Ecuación:  x = ½·(v₀+vf)·t",
                      f"  x = 0.5·({v0}+{vf})·{t}",
                      f"  x = {0.5*(v0+vf)*t:.2f} m"]
            a = (vf-v0)/t
            x = 0.5*(v0+vf)*t

        elif v0 is not None and vf is not None and a is not None \
                and t is None and x is None:
            if a == 0:
                return None, "Si a=0 el tiempo es indeterminado.", []
            steps += ["Ecuación:  t = (vf − v₀) / a",
                      f"  t = ({vf} − {v0}) / {a}",
                      f"  t = {(vf-v0)/a:.2f} s"]
            t = (vf-v0)/a
            x = 0.5*(v0+vf)*t
            steps += ["Ecuación:  x = ½·(v₀+vf)·t",
                      f"  x = 0.5·({v0}+{vf})·{t:.2f}",
                      f"  x = {x:.2f} m"]

        elif v0 is not None and a is not None and x is not None \
                and vf is None and t is None:
            disc = v0**2 + 2*a*x
            if disc < 0:
                return None, "Discriminante negativo — sin solución real.", []
            steps += ["Ecuación:  vf² = v₀² + 2·a·x",
                      f"  vf² = {v0}² + 2·{a}·{x}  =  {disc:.2f}",
                      f"  vf  = √{disc:.2f}  =  {math.sqrt(disc):.2f} m/s"]
            vf = math.sqrt(disc)
            t  = (vf-v0)/a if a != 0 else (x/v0 if v0 != 0 else None)
            if t is not None:
                steps += [f"  t   = (vf−v₀)/a  =  {t:.2f} s"]

        elif vf is not None and a is not None and t is not None \
                and v0 is None and x is None:
            steps += ["Ecuación:  v₀ = vf − a·t",
                      f"  v₀ = {vf} − {a}·{t}",
                      f"  v₀ = {vf-a*t:.2f} m/s"]
            v0 = vf-a*t
            x  = v0*t + 0.5*a*t**2
            steps += ["Ecuación:  x = v₀·t + ½·a·t²",
                      f"  x = {v0:.2f}·{t} + 0.5·{a}·{t}²",
                      f"  x = {x:.2f} m"]

        elif v0 is not None and vf is not None and x is not None \
                and a is None and t is None:
            steps += ["Ecuación:  a = (vf² − v₀²) / (2·x)",
                      f"  a = ({vf}² − {v0}²) / (2·{x})",
                      f"  a = {(vf**2-v0**2)/(2*x):.2f} m/s²"]
            a = (vf**2-v0**2)/(2*x)
            t = 2*x/(v0+vf) if (v0+vf) != 0 else None
            if t:
                steps += ["Ecuación:  t = 2x / (v₀+vf)",
                          f"  t = 2·{x} / ({v0}+{vf})",
                          f"  t = {t:.2f} s"]

        elif a is not None and t is not None and x is not None \
                and v0 is None and vf is None:
            v0 = (x - 0.5*a*t**2)/t if t != 0 else None
            if v0 is not None: vf = v0+a*t
            steps += ["Ecuación:  v₀ = (x − ½·a·t²) / t",
                      f"  v₀ = ({x} − 0.5·{a}·{t}²) / {t}",
                      f"  v₀ = {v0:.2f} m/s",
                      "Ecuación:  vf = v₀ + a·t",
                      f"  vf = {v0:.2f} + {a}·{t}",
                      f"  vf = {vf:.2f} m/s"]

        elif vf is not None and t is not None and x is not None \
                and v0 is None and a is None:
            v0 = 2*x/t - vf if t != 0 else None
            if v0 is not None: a = (vf-v0)/t
            steps += ["Ecuación:  v₀ = 2x/t − vf",
                      f"  v₀ = 2·{x}/{t} − {vf}",
                      f"  v₀ = {v0:.2f} m/s",
                      "Ecuación:  a = (vf − v₀) / t",
                      f"  a = ({vf} − {v0:.2f}) / {t}",
                      f"  a = {a:.2f} m/s²"]

        elif v0 is not None and t is not None and x is not None \
                and a is None and vf is None:
            a = 2*(x-v0*t)/t**2 if t != 0 else None
            if a is not None: vf = v0+a*t
            steps += ["Ecuación:  a = 2·(x − v₀·t) / t²",
                      f"  a = 2·({x} − {v0}·{t}) / {t}²",
                      f"  a = {a:.2f} m/s²",
                      "Ecuación:  vf = v₀ + a·t",
                      f"  vf = {v0:.2f} + {a:.2f}·{t}",
                      f"  vf = {vf:.2f} m/s"]

        else:
            if v0 is not None and a is not None and t is not None:
                if vf is None: vf = v0+a*t
                if x  is None: x  = v0*t+0.5*a*t**2
            elif v0 is not None and vf is not None and t is not None:
                if a  is None: a  = (vf-v0)/t
                if x  is None: x  = 0.5*(v0+vf)*t
            steps += ["Sistema con datos suficientes — verificando."]

        # Línea de verificación
        if v0r := v0:
            pass
        steps += ["",
                  "✔ Verificación:",
                  f"  vf = v₀ + a·t  →  {v0:.2f} + {a:.2f}·{t:.2f} = {v0+a*t:.2f} m/s",
                  f"  x  = v₀·t+½a·t²  →  {x:.2f} m"]

        return {"v0": v0, "vf": vf, "a": a, "t": t, "x": x}, None, steps
    except Exception as e:
        return None, f"Error: {e}", []


def resolver_parabolico(v0=None, angulo=None, x_max=None, y_max=None):
    g = 9.8
    steps = []
    try:
        if v0 is not None and angulo is not None:
            rad = math.radians(angulo)
            v0x = v0*math.cos(rad); v0y = v0*math.sin(rad)
            t_v = 2*v0y/g;  x_m = v0x*t_v;  y_m = v0y**2/(2*g)
            steps += [
                "Descomponer v₀ en componentes:",
                f"  v₀ₓ = v₀·cos(θ) = {v0}·cos({angulo}°) = {v0x:.2f} m/s",
                f"  v₀ᵧ = v₀·sin(θ) = {v0}·sin({angulo}°) = {v0y:.2f} m/s",
                "Tiempo de vuelo:  T = 2·v₀ᵧ / g",
                f"  T = 2·{v0y:.2f} / 9.8 = {t_v:.2f} s",
                "Alcance máximo:  xₘₐₓ = v₀ₓ · T",
                f"  xₘₐₓ = {v0x:.2f} · {t_v:.2f} = {x_m:.2f} m",
                "Altura máxima:  yₘₐₓ = v₀ᵧ² / (2g)",
                f"  yₘₐₓ = {v0y:.2f}² / (2·9.8) = {y_m:.2f} m",
                "",
                "✔ Verificación:",
                f"  xₘₐₓ = v₀²·sin(2θ)/g = {v0**2*math.sin(2*rad)/g:.2f} m",
            ]
            return {"v0":v0,"angulo":angulo,"t_vuelo":t_v,
                    "x_max":x_m,"y_max":y_m,"v0x":v0x,"v0y":v0y}, None, steps

        elif x_max is not None and angulo is not None:
            rad = math.radians(angulo)
            v0c = math.sqrt(x_max*g/math.sin(2*rad))
            v0x = v0c*math.cos(rad); v0y = v0c*math.sin(rad)
            t_v = 2*v0y/g;  y_m = v0y**2/(2*g)
            steps += [
                "Despejar v₀ de:  xₘₐₓ = v₀²·sin(2θ) / g",
                f"  v₀ = √( xₘₐₓ·g / sin(2θ) )",
                f"  v₀ = √( {x_max}·9.8 / sin({2*angulo}°) ) = {v0c:.2f} m/s",
                f"  v₀ₓ = {v0x:.2f} m/s     v₀ᵧ = {v0y:.2f} m/s",
                f"  T   = {t_v:.2f} s        yₘₐₓ = {y_m:.2f} m",
                "",
                "✔ Verificación:",
                f"  xₘₐₓ = v₀²·sin(2θ)/g = {v0c**2*math.sin(2*rad)/g:.2f} m",
            ]
            return {"v0":v0c,"angulo":angulo,"t_vuelo":t_v,
                    "x_max":x_max,"y_max":y_m,"v0x":v0x,"v0y":v0y}, None, steps

        elif y_max is not None and angulo is not None:
            rad = math.radians(angulo)
            v0y = math.sqrt(2*g*y_max)
            v0c = v0y/math.sin(rad)
            v0x = v0c*math.cos(rad)
            t_v = 2*v0y/g;  x_m = v0x*t_v
            steps += [
                "Despejar v₀ᵧ de:  yₘₐₓ = v₀ᵧ² / (2g)",
                f"  v₀ᵧ = √(2·g·yₘₐₓ) = √(2·9.8·{y_max}) = {v0y:.2f} m/s",
                "Obtener v₀:  v₀ = v₀ᵧ / sin(θ)",
                f"  v₀ = {v0y:.2f} / sin({angulo}°) = {v0c:.2f} m/s",
                f"  v₀ₓ = {v0x:.2f} m/s",
                "Tiempo de vuelo:  T = 2·v₀ᵧ / g",
                f"  T = {t_v:.2f} s",
                "Alcance:  xₘₐₓ = v₀ₓ · T",
                f"  xₘₐₓ = {v0x:.2f}·{t_v:.2f} = {x_m:.2f} m",
                "",
                "✔ Verificación:",
                f"  yₘₐₓ = v₀ᵧ²/(2g) = {v0y**2/(2*g):.2f} m",
            ]
            return {"v0":v0c,"angulo":angulo,"t_vuelo":t_v,
                    "x_max":x_m,"y_max":y_max,"v0x":v0x,"v0y":v0y}, None, steps

        elif v0 is not None and x_max is not None:
            val = x_max*g/v0**2
            if abs(val) > 1:
                return None, "Alcance mayor al máximo posible con esa v₀.", []
            ang = math.degrees(0.5*math.asin(val))
            rad = math.radians(ang)
            v0x = v0*math.cos(rad); v0y = v0*math.sin(rad)
            t_v = 2*v0y/g;  y_m = v0y**2/(2*g)
            steps += [
                "Despejar θ de:  xₘₐₓ = v₀²·sin(2θ) / g",
                f"  sin(2θ) = xₘₐₓ·g / v₀² = {x_max}·9.8 / {v0}² = {val:.2f}",
                f"  θ = ½·arcsin({val:.2f}) = {ang:.2f}°",
                f"  v₀ₓ = {v0x:.2f} m/s     v₀ᵧ = {v0y:.2f} m/s",
                f"  T   = {t_v:.2f} s        yₘₐₓ = {y_m:.2f} m",
                "",
                "✔ Verificación:",
                f"  xₘₐₓ = v₀²·sin(2θ)/g = {v0**2*math.sin(2*rad)/g:.2f} m",
            ]
            return {"v0":v0,"angulo":ang,"t_vuelo":t_v,
                    "x_max":x_max,"y_max":y_m,"v0x":v0x,"v0y":v0y}, None, steps

        return None, "Combinación de datos insuficiente.", []
    except Exception as e:
        return None, f"Error: {e}", []


# ═══════════════════════════════════════════════════════════════════
#  WIDGET: ENTRY CON VALIDACIÓN VISUAL
# ═══════════════════════════════════════════════════════════════════

class ValidatedEntry(tk.Frame):
    def __init__(self, parent, label, unit="",
                 allow_negative=True, allow_zero=True, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.allow_negative = allow_negative
        self.allow_zero     = allow_zero
        self._valid         = True

        row = tk.Frame(self, bg=PANEL)
        row.pack(fill="x")

        self.bar = tk.Frame(row, bg=BORDER, width=3)
        self.bar.pack(side="left", fill="y", padx=(0, 5))

        inner = tk.Frame(row, bg=PANEL)
        inner.pack(fill="x", expand=True)

        head = tk.Frame(inner, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text=label, bg=PANEL, fg=TEXT,
                 font=(FM, 11), anchor="w").pack(side="left")
        if unit:
            tk.Label(head, text=f" [{unit}]", bg=PANEL, fg=SUBTEXT,
                     font=(FM, 8)).pack(side="left")
        self.status = tk.Label(head, text="— incógnita", bg=PANEL,
                               fg=SUBTEXT, font=(FM, 8))
        self.status.pack(side="right")

        self.var = tk.StringVar()
        self.ent = tk.Entry(inner, textvariable=self.var,
                            bg=BG_ENTRY, fg=TEXT, font=(FM, 11),
                            relief="flat", bd=0, insertbackground=ACCENT,
                            highlightthickness=1, highlightcolor=ACCENT,
                            highlightbackground=BORDER)
        self.ent.pack(fill="x", ipady=4)
        self.var.trace_add("write", self._validate)

    def _validate(self, *_):
        s = self.var.get().strip()
        if s in ("", "?"):
            self._set_state("empty"); return
        try:
            v = float(s)
            if not self.allow_negative and v < 0:
                self._set_state("error", "⚠ no negativo"); return
            if not self.allow_zero and v == 0:
                self._set_state("error", "⚠ no puede ser 0"); return
            self._set_state("ok")
        except ValueError:
            self._set_state("error", "⚠ número inválido")

    def _set_state(self, state, msg=""):
        if state == "ok":
            self.bar.config(bg=SUCCESS)
            self.status.config(text="✓ válido", fg=SUCCESS)
            self.ent.config(highlightbackground=SUCCESS)
            self._valid = True
        elif state == "error":
            self.bar.config(bg=ERROR)
            self.status.config(text=msg, fg=ERROR)
            self.ent.config(highlightbackground=ERROR)
            self._valid = False
        else:
            self.bar.config(bg=BORDER)
            self.status.config(text="— incógnita", fg=SUBTEXT)
            self.ent.config(highlightbackground=BORDER)
            self._valid = True

    def get_value(self):
        s = self.var.get().strip()
        if s in ("", "?"): return None
        try: return float(s)
        except: return None

    def set(self, v): self.var.set(v)
    def is_valid(self): return self._valid


# ═══════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

class SimuladorFisica:
    def __init__(self, root):
        self.root = root
        self.root.title("Física I — Simulador v2.1")
        self.root.configure(bg=BG)
        self.root.geometry("1400x880")
        self.root.minsize(1100, 720)

        self._anim       = None
        self._animando   = False
        self._anim_pts   = []
        self._anim_frame = 0
        self._anim_step  = 1
        self._anim_N     = 0

        self._build_ui()

    # ══════════════════════════════════════
    #  UI PRINCIPAL
    # ══════════════════════════════════════
    def _build_ui(self):
        # ── Header ──────────────────────
        hdr = tk.Frame(self.root, bg=PANEL2, height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚛  FÍSICA I", bg=PANEL2, fg=ACCENT,
                 font=(FM, 20, "bold")).pack(side="left", padx=28, pady=10)
        tk.Label(hdr, text="Simulador · Cinemática · Dinámica · Resolutor",
                 bg=PANEL2, fg=SUBTEXT, font=(FM, 11)).pack(side="left")
        tk.Label(hdr, text=" v2.1 ", bg=ACCENT, fg=BG,
                 font=(FM, 9, "bold"), padx=6, pady=2).pack(side="right", padx=20)

        # ── Body: sidebar + contenido ───
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Sidebar
        side = tk.Frame(main, bg=SIDEBAR, width=210)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="MÓDULOS", bg=SIDEBAR, fg=SUBTEXT,
                 font=(FM, 9, "bold")).pack(pady=(24, 6), padx=20, anchor="w")
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16)

        self.btn_tabs = {}
        for lbl, key in [("📐  MRUA", "mrua"),
                          ("🎯  Parabólico", "parabolico"),
                          ("⚡  Newton", "newton")]:
            b = tk.Button(side, text=lbl, bg=SIDEBAR, fg=TEXT,
                          font=(FM, 12), relief="flat", bd=0,
                          anchor="w", padx=20, pady=12, cursor="hand2",
                          activebackground=PANEL2, activeforeground=ACCENT,
                          command=lambda k=key: self._switch_tab(k))
            b.pack(fill="x")
            self.btn_tabs[key] = b

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        tk.Label(side, text="💡 Deja en blanco\nlas incógnitas",
                 bg=SIDEBAR, fg=SUBTEXT, font=(FM, 9),
                 justify="center").pack(pady=4)

        # Barra de progreso simulación
        tk.Label(side, text="SIMULACIÓN", bg=SIDEBAR, fg=SUBTEXT,
                 font=(FM, 8)).pack(pady=(14, 2), padx=16, anchor="w")
        self.prog_canvas = tk.Canvas(side, bg=BORDER, height=5,
                                     highlightthickness=0)
        self.prog_canvas.pack(fill="x", padx=10)
        self.prog_bar = self.prog_canvas.create_rectangle(
            0, 0, 0, 5, fill=ACCENT, width=0)

        # ── Contenido: panel izquierdo + derecho ──
        content = tk.Frame(main, bg=BG)
        content.pack(side="left", fill="both", expand=True)

        # Panel izquierdo (formulario + solución)
        self.left_panel = tk.Frame(content, bg=PANEL, width=430)
        self.left_panel.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self.left_panel.pack_propagate(False)

        # Panel derecho (gráficas)
        self.right_panel = tk.Frame(content, bg=PANEL)
        self.right_panel.pack(side="left", fill="both", expand=True,
                               padx=(6, 12), pady=12)

        # ── Gráficas ─────────────────────
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(8, 5), facecolor=PANEL)
        self.fig.subplots_adjust(left=0.10, right=0.97,
                                  top=0.88, bottom=0.12,
                                  hspace=0.48, wspace=0.38)

        self.ax_main = self.fig.add_subplot(2, 2, 1)
        self.ax_vel  = self.fig.add_subplot(2, 2, 2)
        self.ax_acel = self.fig.add_subplot(2, 2, 3)
        self.ax_rt   = self.fig.add_subplot(2, 2, 4)

        for ax in [self.ax_main, self.ax_vel, self.ax_acel, self.ax_rt]:
            ax.set_facecolor(BG_ENTRY)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(alpha=0.15, color=BORDER)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Stats bar bajo gráficas
        stats = tk.Frame(self.right_panel, bg=BG_ENTRY, height=36)
        stats.pack(fill="x", padx=6, pady=(0, 6))
        stats.pack_propagate(False)
        self.stats_labels = {}
        for key, lbl in [("v_max", "V_MAX"), ("x_max", "X_MAX"),
                          ("t_total", "T_TOTAL"), ("a_net", "A_NET")]:
            f = tk.Frame(stats, bg=BG_ENTRY)
            f.pack(side="left", expand=True, fill="both")
            tk.Frame(f, bg=BORDER, width=1).pack(side="left", fill="y")
            inner = tk.Frame(f, bg=BG_ENTRY)
            inner.pack(expand=True)
            tk.Label(inner, text=lbl, bg=BG_ENTRY, fg=SUBTEXT,
                     font=(FM, 7)).pack()
            v = tk.Label(inner, text="—", bg=BG_ENTRY, fg=ACCENT,
                         font=(FM, 9, "bold"))
            v.pack()
            self.stats_labels[key] = v

        # ── Panel de solución (dentro de left_panel, parte inferior) ──
        self._build_solution_panel()

        # ── Tabs de formulario ────────────
        self.tab_frames = {}
        self._build_mrua()
        self._build_parabolico()
        self._build_newton()
        self._switch_tab("mrua")

    # ══════════════════════════════════════
    #  PANEL DE SOLUCIÓN (parte baja del left_panel)
    # ══════════════════════════════════════
    def _build_solution_panel(self):
        """Panel fijo en la parte inferior del panel izquierdo."""
        # Separador
        tk.Frame(self.left_panel, bg=BORDER, height=1).pack(
            fill="x", side="bottom", padx=8, pady=0)

        # Contenedor principal de solución (pack desde abajo)
        self.sol_container = tk.Frame(self.left_panel, bg=PANEL,
                                       height=260)
        self.sol_container.pack(side="bottom", fill="x", padx=0)
        self.sol_container.pack_propagate(False)

        # Cabecera del panel de solución
        sol_hdr = tk.Frame(self.sol_container, bg=ACCENT, height=34)
        sol_hdr.pack(fill="x")
        sol_hdr.pack_propagate(False)
        tk.Label(sol_hdr, text="  ▼ SOLUCIÓN DE ECUACIONES",
                 bg=ACCENT, fg=BG, font=(FM, 9, "bold")).pack(
                     side="left", pady=7)
        # Botón Limpiar permanente — siempre visible
        tk.Button(sol_hdr, text="↺  Limpiar",
                  bg=BG, fg=ACCENT, font=(FM, 8, "bold"),
                  relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2",
                  activebackground=PANEL2, activeforeground=ACCENT,
                  command=self._limpiar_todo).pack(side="right", padx=6, pady=5)
        self.sol_unk_lbl = tk.Label(sol_hdr, text="",
                                     bg=ACCENT, fg=BG, font=(FM, 8))
        self.sol_unk_lbl.pack(side="right", padx=4)

        # Área de texto con scroll
        txt_wrap = tk.Frame(self.sol_container, bg=BG_ENTRY)
        txt_wrap.pack(fill="both", expand=True)

        vsb = tk.Scrollbar(txt_wrap, orient="vertical",
                           bg=PANEL2, troughcolor=PANEL,
                           width=8)
        vsb.pack(side="right", fill="y")

        self.sol_text = tk.Text(txt_wrap, bg=BG_ENTRY, fg=TEXT,
                                 font=(FM, 9), relief="flat", bd=0,
                                 wrap="word", state="disabled",
                                 yscrollcommand=vsb.set,
                                 padx=10, pady=6,
                                 spacing1=1, spacing2=1,
                                 selectbackground=PANEL2,
                                 insertbackground=ACCENT)
        self.sol_text.pack(fill="both", expand=True)
        vsb.config(command=self.sol_text.yview)

        # Tags de color
        self.sol_text.tag_config("titulo",
                                  foreground=ACCENT,
                                  font=(FM, 9, "bold"))
        self.sol_text.tag_config("ecuacion",
                                  foreground=ACCENT_YEL,
                                  font=(FM, 9, "italic"))
        self.sol_text.tag_config("sustitucion",
                                  foreground=TEXT,
                                  font=(FM, 9))
        self.sol_text.tag_config("resultado",
                                  foreground=SUCCESS,
                                  font=(FM, 9, "bold"))
        self.sol_text.tag_config("verif",
                                  foreground=ACCENT_BLU,
                                  font=(FM, 9, "italic"))
        self.sol_text.tag_config("sep",
                                  foreground=BORDER,
                                  font=(FM, 7))

    # ── Escribir solución en el panel ────
    def _show_solution(self, pasos, incognitas=""):
        self.sol_unk_lbl.configure(
            text=f"Calculado: {incognitas}" if incognitas else "")

        txt = self.sol_text
        txt.configure(state="normal")
        txt.delete("1.0", "end")

        for paso in pasos:
            if paso == "":
                txt.insert("end", "\n", "sustitucion")
            elif paso.startswith("✔"):
                txt.insert("end", paso + "\n", "titulo")
            elif paso.startswith("Ecuación") or paso.startswith("Despejar") \
                    or paso.startswith("Descompon") or paso.startswith("Obtener") \
                    or paso.startswith("Tiempo") or paso.startswith("Alcance") \
                    or paso.startswith("Altura") or paso.startswith("Fuerza") \
                    or paso.startswith("Segunda") or paso.startswith("Datos"):
                txt.insert("end", paso + "\n", "ecuacion")
            elif paso.strip().startswith("  ") and "=" in paso:
                # líneas con resultado numérico
                if paso.strip().startswith("  ") and \
                        any(u in paso for u in ["m/s", "m/s²", " m", " s", " N", " kg"]):
                    txt.insert("end", paso + "\n", "resultado")
                else:
                    txt.insert("end", paso + "\n", "sustitucion")
            elif "✔" in paso or "Verificación" in paso:
                txt.insert("end", paso + "\n", "verif")
            else:
                txt.insert("end", paso + "\n", "sustitucion")

        txt.configure(state="disabled")
        txt.see("1.0")

    def _clear_solution(self):
        self.sol_text.configure(state="normal")
        self.sol_text.delete("1.0", "end")
        self.sol_text.configure(state="disabled")
        self.sol_unk_lbl.configure(text="")

    def _limpiar_todo(self):
        """Limpia formulario, gráficas y solución del módulo activo."""
        self._stop_animation()
        # Detectar tab activo y limpiar sus campos
        activo = None
        for k, f in self.tab_frames.items():
            try:
                if f.winfo_ismapped():
                    activo = k
                    break
            except Exception:
                pass
        if activo == "mrua":
            for e in [self.m_v0, self.m_vf, self.m_a, self.m_t, self.m_x]:
                e.set("")
        elif activo == "parabolico":
            for e in [self.p_v0, self.p_ang, self.p_xmax, self.p_ymax]:
                e.set("")
        elif activo == "newton":
            for e in [self.n_m, self.n_f, self.n_ff, self.n_a]:
                e.set("")
            self.n_ff.set("0")
            self.n_t.set("8")
        self._clear_graphs()
        self._clear_solution()

    # ══════════════════════════════════════
    #  SWITCH / LIMPIAR
    # ══════════════════════════════════════
    def _switch_tab(self, key):
        self._stop_animation()
        for k, f in self.tab_frames.items():
            f.pack_forget()
        for k, b in self.btn_tabs.items():
            b.configure(bg=SIDEBAR, fg=TEXT)
        self.tab_frames[key].pack(fill="x", padx=16, pady=8)
        self.btn_tabs[key].configure(bg=PANEL2, fg=ACCENT)
        self._clear_graphs()
        self._clear_solution()

    def _stop_animation(self):
        self._animando = False
        if self._anim is not None:
            try: self.root.after_cancel(self._anim)
            except: pass
            self._anim = None
        self._anim_pts   = []
        self._anim_frame = 0
        try: self.prog_canvas.coords(self.prog_bar, 0, 0, 0, 5)
        except: pass

    def _clear_graphs(self):
        for ax in [self.ax_main, self.ax_vel, self.ax_acel, self.ax_rt]:
            ax.cla()
            ax.set_facecolor(BG_ENTRY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(alpha=0.15, color=BORDER)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
        self.canvas.draw()
        for lb in self.stats_labels.values():
            lb.configure(text="—")

    def _update_stats(self, v_max=None, x_max=None,
                      t_total=None, a_net=None):
        units = {"v_max": "m/s", "x_max": "m",
                 "t_total": "s", "a_net": "m/s²"}
        for k, v in [("v_max", v_max), ("x_max", x_max),
                      ("t_total", t_total), ("a_net", a_net)]:
            self.stats_labels[k].configure(
                text=f"{v:.3f} {units[k]}" if v is not None else "—")

    # ══════════════════════════════════════
    #  GRÁFICAS
    # ══════════════════════════════════════
    def _ax_style(self, ax, title, xl, yl):
        ax.set_title(title, color=TEXT, fontsize=8, fontfamily=FM)
        ax.set_xlabel(xl, color=SUBTEXT, fontsize=7)
        ax.set_ylabel(yl, color=SUBTEXT, fontsize=7)

    def _plot_full(self, datos, modo, titulo):
        for ax in [self.ax_main, self.ax_vel, self.ax_acel, self.ax_rt]:
            ax.cla()
            ax.set_facecolor(BG_ENTRY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(alpha=0.15, color=BORDER)
            ax.tick_params(colors=SUBTEXT, labelsize=8)

        self.fig.suptitle(titulo, color=ACCENT, fontsize=11,
                          fontfamily=FM, fontweight="bold")
        t = datos["t"]

        if modo in ("mrua", "newton"):
            x = datos["x"]; v = datos["v"]
            a_arr = datos.get("a", np.full_like(t, datos.get("ax", 0)))

            self.ax_main.plot(t, x, color=LINE1, lw=2)
            self.ax_main.fill_between(t, 0, x, alpha=0.08, color=LINE1)
            self._ax_style(self.ax_main, "Posición x(t)", "t (s)", "x (m)")

            self.ax_vel.plot(t, v, color=LINE2, lw=2)
            self.ax_vel.fill_between(t, 0, v, alpha=0.08, color=LINE2)
            self._ax_style(self.ax_vel, "Velocidad v(t)", "t (s)", "v (m/s)")

            self.ax_acel.plot(t, a_arr, color=LINE3, lw=2, ls="--")
            self.ax_acel.fill_between(t, 0, a_arr, alpha=0.08, color=LINE3)
            self._ax_style(self.ax_acel, "Aceleración a(t)", "t (s)", "a (m/s²)")

            self.ax_rt.plot(x, v, color=ACCENT_BLU, lw=2)
            self._ax_style(self.ax_rt, "Fase v(x)", "x (m)", "v (m/s)")

            pt1, = self.ax_main.plot([], [], "o", color=ACCENT,     ms=7, zorder=5)
            pt2, = self.ax_vel.plot([], [], "o",  color=LINE2,      ms=7, zorder=5)
            pt3, = self.ax_acel.plot([], [], "o", color=LINE3,      ms=7, zorder=5)
            pt4, = self.ax_rt.plot([], [], "o",   color=ACCENT_BLU, ms=7, zorder=5)
            self._anim_pts = [(pt1,t,x),(pt2,t,v),(pt3,t,a_arr),(pt4,x,v)]

        elif modo == "parabolico":
            x = datos["x"]; y = datos["y"]; v = datos["v"]
            g_arr = np.full_like(t, -9.8)
            ke    = 0.5 * v**2
            ke_n  = ke / max(ke) if max(ke) > 0 else ke

            self.ax_main.plot(x, y, color=LINE1, lw=2)
            self.ax_main.fill_between(x, 0, y, alpha=0.08, color=LINE1)
            im = np.argmax(y)
            self.ax_main.plot(x[im], y[im], "*", color=ACCENT_YEL, ms=11, zorder=5)
            self.ax_main.annotate(
                f"yₘₐₓ={y[im]:.1f}m", (x[im], y[im]),
                xytext=(4, 4), textcoords="offset points",
                color=ACCENT_YEL, fontsize=7, fontfamily=FM)
            self._ax_style(self.ax_main, "Trayectoria y(x)", "x (m)", "y (m)")

            self.ax_vel.plot(t, v, color=LINE2, lw=2)
            self.ax_vel.fill_between(t, 0, v, alpha=0.08, color=LINE2)
            self._ax_style(self.ax_vel, "|v(t)|", "t (s)", "|v| (m/s)")

            self.ax_acel.plot(t, g_arr, color=LINE3, lw=2, ls="--")
            self._ax_style(self.ax_acel, "Acel. vert. (−g)", "t (s)", "aᵧ (m/s²)")

            self.ax_rt.plot(t, ke_n, color=ACCENT_BLU, lw=2)
            self._ax_style(self.ax_rt, "Ec. cinética norm.", "t (s)", "Ec/Ecₘₐₓ")

            pt1, = self.ax_main.plot([], [], "o", color=ACCENT,     ms=7, zorder=5)
            pt2, = self.ax_vel.plot([], [], "o",  color=LINE2,      ms=7, zorder=5)
            pt3, = self.ax_acel.plot([], [], "o", color=LINE3,      ms=7, zorder=5)
            pt4, = self.ax_rt.plot([], [], "o",   color=ACCENT_BLU, ms=7, zorder=5)
            self._anim_pts = [(pt1,x,y),(pt2,t,v),(pt3,t,g_arr),(pt4,t,ke_n)]

        self.canvas.draw()
        self._start_animation(datos)

    def _draw_forces(self, ax, Fx, ff):
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.axis("off"); ax.set_facecolor(BG_ENTRY)
        ax.set_title("Diagrama de fuerzas", color=TEXT,
                     fontsize=8, fontfamily=FM)
        rect = plt.Rectangle((-0.6, -0.6), 1.2, 1.2,
                              color=ACCENT, alpha=0.5, zorder=3)
        ax.add_patch(rect)
        ax.text(0, 0, "m", color=BG, fontsize=11, ha="center",
                va="center", fontweight="bold", zorder=4, fontfamily=FM)
        if Fx != 0:
            sc = min(abs(Fx)/25, 1.3) * (1 if Fx > 0 else -1)
            ax.annotate("", xy=(0.6+sc, 0), xytext=(0.6, 0),
                        arrowprops=dict(arrowstyle="->", color=ACCENT_YEL, lw=2))
            ax.text(0.6+sc+(0.15 if sc > 0 else -0.15), 0.25,
                    f"F={Fx:.1f}N", color=ACCENT_YEL, fontsize=8, fontfamily=FM)
        if ff > 0:
            ax.annotate("", xy=(-0.6-0.8, 0), xytext=(-0.6, 0),
                        arrowprops=dict(arrowstyle="->", color=LINE3, lw=2))
            ax.text(-1.7, -0.28, f"Ff={ff:.1f}N",
                    color=LINE3, fontsize=8, fontfamily=FM)
        ax.annotate("", xy=(0,-1.8), xytext=(0,-0.6),
                    arrowprops=dict(arrowstyle="->", color=LINE3, lw=1.8))
        ax.text(0.12,-1.4,"W=mg", color=LINE3, fontsize=8, fontfamily=FM)
        ax.annotate("", xy=(0, 1.8), xytext=(0, 0.6),
                    arrowprops=dict(arrowstyle="->", color=ACCENT_BLU, lw=1.8))
        ax.text(0.12, 1.3,"N", color=ACCENT_BLU, fontsize=8, fontfamily=FM)

    # ══════════════════════════════════════
    #  ANIMACIÓN
    # ══════════════════════════════════════
    def _start_animation(self, datos):
        self._stop_animation()
        N = len(datos["t"])
        self._anim_step  = max(1, N // 120)
        self._anim_N     = N
        self._anim_frame = 0
        self._animando   = True
        pts = list(self._anim_pts)

        def tick():
            if not self._animando: return
            idx = min(self._anim_frame * self._anim_step, N - 1)
            try:
                for pt, xs, ys in pts:
                    if idx < len(xs) and idx < len(ys):
                        pt.set_data([xs[idx]], [ys[idx]])
                self.canvas.draw_idle()
                w = self.prog_canvas.winfo_width()
                pct = idx / max(N-1, 1)
                self.prog_canvas.coords(self.prog_bar, 0, 0, int(w*pct), 5)
            except Exception:
                self._animando = False; return
            self._anim_frame += 1
            total_frames = math.ceil(N / self._anim_step)
            if self._anim_frame < total_frames:
                self._anim = self.root.after(16, tick)
            else:
                try:
                    for pt, xs, ys in pts:
                        pt.set_data([xs[-1]], [ys[-1]])
                    self.canvas.draw_idle()
                    w = self.prog_canvas.winfo_width()
                    self.prog_canvas.coords(self.prog_bar, 0, 0, w, 5)
                except Exception: pass
                self._anim = None; self._animando = False

        self._anim = self.root.after(16, tick)

    # ══════════════════════════════════════
    #  PESTAÑA: MRUA
    # ══════════════════════════════════════
    def _build_mrua(self):
        f = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["mrua"] = f

        card = tk.Frame(f, bg=PANEL2)
        card.pack(fill="x", pady=(0, 6))
        tk.Label(card, text="📐 MRUA", bg=PANEL2, fg=ACCENT,
                 font=(FM, 13, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        for eq in ["vf = v₀ + a·t",
                   "x  = v₀·t + ½·a·t²",
                   "vf² = v₀² + 2·a·x",
                   "x  = ½·(v₀+vf)·t"]:
            tk.Label(card, text=f"  {eq}", bg=PANEL2, fg=SUBTEXT,
                     font=(FM, 9)).pack(anchor="w", padx=12)
        tk.Label(card, text="", bg=PANEL2).pack(pady=2)

        self.m_v0 = ValidatedEntry(f, "v₀  vel. inicial", unit="m/s")
        self.m_vf = ValidatedEntry(f, "vf  vel. final",   unit="m/s")
        self.m_a  = ValidatedEntry(f, "a   aceleración",  unit="m/s²")
        self.m_t  = ValidatedEntry(f, "t   tiempo",       unit="s",
                                   allow_negative=False, allow_zero=False)
        self.m_x  = ValidatedEntry(f, "x   desplazamiento", unit="m")
        for w in [self.m_v0, self.m_vf, self.m_a, self.m_t, self.m_x]:
            w.pack(fill="x", pady=2)

        self.m_ctr = tk.Label(f, text="0 / mínimo 3 valores",
                               bg=PANEL, fg=SUBTEXT, font=(FM, 9))
        self.m_ctr.pack(anchor="w", pady=(4, 0))
        for e in [self.m_v0, self.m_vf, self.m_a, self.m_t, self.m_x]:
            e.var.trace_add("write", lambda *_: self._upd_mrua())

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._btn_m = tk.Button(f, text="▶  CALCULAR Y SIMULAR",
                                 bg=BORDER, fg=SUBTEXT, font=(FM, 12, "bold"),
                                 relief="flat", bd=0, pady=9, cursor="hand2",
                                 activebackground=ACCENT2, activeforeground=BG,
                                 command=self._sim_mrua)
        self._btn_m.pack(fill="x")

    def _upd_mrua(self):
        c = sum(1 for e in [self.m_v0, self.m_vf, self.m_a,
                              self.m_t, self.m_x]
                if e.get_value() is not None)
        col = SUCCESS if c >= 3 else WARN if c == 2 else SUBTEXT
        self.m_ctr.configure(text=f"{c} / mínimo 3 valores", fg=col)
        self._btn_m.configure(bg=ACCENT if c >= 3 else BORDER,
                               fg=BG    if c >= 3 else SUBTEXT)

    def _limpiar_mrua(self):
        self._stop_animation()
        for e in [self.m_v0, self.m_vf, self.m_a, self.m_t, self.m_x]:
            e.set("")
        self._clear_graphs(); self._clear_solution()

    def _sim_mrua(self):
        self._stop_animation()
        if not all(e.is_valid() for e in [self.m_v0, self.m_vf,
                                            self.m_a, self.m_t, self.m_x]):
            messagebox.showerror("Validación", "Corrige los campos en rojo.")
            return

        v0 = self.m_v0.get_value(); vf = self.m_vf.get_value()
        a  = self.m_a.get_value();  t  = self.m_t.get_value()
        x  = self.m_x.get_value()

        res, err, steps = resolver_mrua(v0, vf, a, t, x)
        if err:
            messagebox.showerror("Error", err); return

        v0r, vfr, ar, tr, xr = (res["v0"], res["vf"],
                                  res["a"],  res["t"], res["x"])
        if any(v is None for v in [v0r, vfr, ar, tr, xr]):
            messagebox.showwarning("Incompleto",
                "No se pudieron resolver todas las incógnitas."); return
        if tr <= 0:
            messagebox.showwarning("Advertencia",
                "El tiempo debe ser positivo."); return

        solved = [n for n, v in [("v₀", v0), ("vf", vf), ("a", a),
                                   ("t", t), ("x", x)] if v is None]
        self._show_solution(steps, ", ".join(solved) if solved else "")

        datos = cinematica_mrua(v0r, ar, tr)
        self._plot_full(datos, "mrua",
                        f"MRUA  v₀={v0r:.2f}  a={ar:.2f}  t={tr:.2f}s")
        self._update_stats(v_max=float(np.max(np.abs(datos["v"]))),
                           x_max=float(np.max(datos["x"])),
                           t_total=tr, a_net=ar)

    # ══════════════════════════════════════
    #  PESTAÑA: TIRO PARABÓLICO
    # ══════════════════════════════════════
    def _build_parabolico(self):
        f = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["parabolico"] = f

        card = tk.Frame(f, bg=PANEL2)
        card.pack(fill="x", pady=(0, 6))
        tk.Label(card, text="🎯 Tiro Parabólico", bg=PANEL2, fg=ACCENT,
                 font=(FM, 13, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        for eq in ["x   = v₀·cos(θ)·t",
                   "y   = v₀·sin(θ)·t − ½g·t²",
                   "xₘₐₓ = v₀²·sin(2θ)/g",
                   "yₘₐₓ = v₀²·sin²(θ)/(2g)",
                   "T   = 2·v₀·sin(θ)/g    g=9.8 m/s²"]:
            tk.Label(card, text=f"  {eq}", bg=PANEL2, fg=SUBTEXT,
                     font=(FM, 9)).pack(anchor="w", padx=12)
        tk.Label(card, text="", bg=PANEL2).pack(pady=2)

        self.p_v0   = ValidatedEntry(f, "v₀   vel. inicial", unit="m/s",
                                     allow_negative=False, allow_zero=False)
        self.p_ang  = ValidatedEntry(f, "θ    ángulo",       unit="°")
        self.p_xmax = ValidatedEntry(f, "xₘₐₓ  alcance",    unit="m",
                                     allow_negative=False)
        self.p_ymax = ValidatedEntry(f, "yₘₐₓ  alt. máx",   unit="m",
                                     allow_negative=False)
        for w in [self.p_v0, self.p_ang, self.p_xmax, self.p_ymax]:
            w.pack(fill="x", pady=2)

        sl_row = tk.Frame(f, bg=PANEL)
        sl_row.pack(fill="x", pady=(4, 0))
        tk.Label(sl_row, text="θ:", bg=PANEL, fg=SUBTEXT,
                 font=(FM, 9)).pack(side="left")
        sl = tk.Scale(sl_row, from_=1, to=89, orient="horizontal",
                      bg=PANEL, fg=TEXT, troughcolor=BORDER,
                      activebackground=ACCENT, highlightthickness=0,
                      bd=0, font=(FM, 8),
                      command=lambda v: self.p_ang.set(str(v)))
        sl.set(45)
        sl.pack(side="left", fill="x", expand=True)

        self.p_ctr = tk.Label(f, text="0 / mínimo 2 valores",
                               bg=PANEL, fg=SUBTEXT, font=(FM, 9))
        self.p_ctr.pack(anchor="w", pady=(4, 0))
        for e in [self.p_v0, self.p_ang, self.p_xmax, self.p_ymax]:
            e.var.trace_add("write", lambda *_: self._upd_par())

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._btn_p = tk.Button(f, text="▶  CALCULAR Y SIMULAR",
                                 bg=BORDER, fg=SUBTEXT, font=(FM, 12, "bold"),
                                 relief="flat", bd=0, pady=9, cursor="hand2",
                                 activebackground=ACCENT2, activeforeground=BG,
                                 command=self._sim_par)
        self._btn_p.pack(fill="x")

    def _upd_par(self):
        c = sum(1 for e in [self.p_v0, self.p_ang,
                              self.p_xmax, self.p_ymax]
                if e.get_value() is not None)
        col = SUCCESS if c >= 2 else WARN if c == 1 else SUBTEXT
        self.p_ctr.configure(text=f"{c} / mínimo 2 valores", fg=col)
        self._btn_p.configure(bg=ACCENT if c >= 2 else BORDER,
                               fg=BG    if c >= 2 else SUBTEXT)

    def _sim_par(self):
        self._stop_animation()
        if not all(e.is_valid() for e in [self.p_v0, self.p_ang,
                                            self.p_xmax, self.p_ymax]):
            messagebox.showerror("Validación", "Corrige los campos en rojo.")
            return

        v0  = self.p_v0.get_value()
        ang = self.p_ang.get_value()
        xm  = self.p_xmax.get_value()
        ym  = self.p_ymax.get_value()

        res, err, steps = resolver_parabolico(v0, ang, xm, ym)
        if err:
            messagebox.showerror("Error", err); return

        v0r  = res["v0"];    angr = res["angulo"]
        xmr  = res["x_max"]; ymr  = res["y_max"]
        tvr  = res["t_vuelo"]

        if not (0 < angr < 90):
            messagebox.showwarning("Advertencia",
                "El ángulo debe estar entre 0° y 90°."); return

        self._show_solution(steps)
        datos = cinematica_mrua(v0r, 0, tvr*1.02, angulo=angr)
        self._plot_full(datos, "parabolico",
                        f"Parabólico  v₀={v0r:.2f} m/s  θ={angr:.1f}°")
        self._update_stats(v_max=float(np.max(datos["v"])),
                           x_max=xmr, t_total=tvr, a_net=9.8)

    # ══════════════════════════════════════
    #  PESTAÑA: 2ª LEY DE NEWTON
    # ══════════════════════════════════════
    def _build_newton(self):
        f = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["newton"] = f

        card = tk.Frame(f, bg=PANEL2)
        card.pack(fill="x", pady=(0, 6))
        tk.Label(card, text="⚡ 2ª Ley de Newton", bg=PANEL2, fg=ACCENT,
                 font=(FM, 13, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        for eq in ["ΣF = m · a",
                   "Fnet = F − Ff",
                   "a  = Fnet / m",
                   "m  = Fnet / a"]:
            tk.Label(card, text=f"  {eq}", bg=PANEL2, fg=SUBTEXT,
                     font=(FM, 9)).pack(anchor="w", padx=12)
        tk.Label(card, text="  💡 Deja vacía la incógnita (m, F o a)",
                 bg=PANEL2, fg=ACCENT_YEL,
                 font=(FM, 9, "italic")).pack(anchor="w", padx=12, pady=(2, 6))

        self.n_m  = ValidatedEntry(f, "m    masa",            unit="kg",
                                   allow_negative=False, allow_zero=False)
        self.n_f  = ValidatedEntry(f, "F    fuerza aplicada", unit="N")
        self.n_ff = ValidatedEntry(f, "Ff   rozamiento",      unit="N",
                                   allow_negative=False)
        self.n_a  = ValidatedEntry(f, "a    aceleración",     unit="m/s²")
        self.n_t  = ValidatedEntry(f, "t    tiempo simul.",   unit="s",
                                   allow_negative=False, allow_zero=False)
        self.n_ff.set("0"); self.n_t.set("8")
        for w in [self.n_m, self.n_f, self.n_ff, self.n_a, self.n_t]:
            w.pack(fill="x", pady=2)

        self.n_ctr = tk.Label(f, text="Necesitas ≥2 de: m, F, a",
                               bg=PANEL, fg=SUBTEXT, font=(FM, 9))
        self.n_ctr.pack(anchor="w", pady=(4, 0))
        for e in [self.n_m, self.n_f, self.n_a]:
            e.var.trace_add("write", lambda *_: self._upd_newt())

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._btn_n = tk.Button(f, text="▶  CALCULAR Y SIMULAR",
                                 bg=BORDER, fg=SUBTEXT, font=(FM, 12, "bold"),
                                 relief="flat", bd=0, pady=9, cursor="hand2",
                                 activebackground=ACCENT2, activeforeground=BG,
                                 command=self._sim_newton)
        self._btn_n.pack(fill="x")

    def _upd_newt(self):
        c = sum(1 for e in [self.n_m, self.n_f, self.n_a]
                if e.get_value() is not None)
        col = SUCCESS if c >= 2 else WARN if c == 1 else SUBTEXT
        self.n_ctr.configure(
            text=f"{c}/3 de m, F, a — necesitas ≥2", fg=col)
        self._btn_n.configure(bg=ACCENT if c >= 2 else BORDER,
                               fg=BG    if c >= 2 else SUBTEXT)

    def _sim_newton(self):
        self._stop_animation()
        if not all(e.is_valid() for e in [self.n_m, self.n_f,
                                            self.n_ff, self.n_a, self.n_t]):
            messagebox.showerror("Validación", "Corrige los campos en rojo.")
            return

        m  = self.n_m.get_value()
        fx = self.n_f.get_value()
        ff = self.n_ff.get_value() or 0.0
        a  = self.n_a.get_value()
        tm = self.n_t.get_value() or 8.0
        solved = []

        if m is None and fx is not None and a is not None:
            Fnet = fx - ff
            if a == 0:
                messagebox.showerror("Error",
                    "Si a=0 la masa es indeterminada."); return
            m = Fnet / a; solved.append("m")
        elif fx is None and m is not None and a is not None:
            fx = m * a + ff; solved.append("F")
        elif a is None and m is not None and fx is not None:
            a = (fx - ff) / m; solved.append("a")
        elif m is not None and fx is not None:
            a = (fx - ff) / m
        else:
            messagebox.showerror("Error",
                "Ingresa ≥2 de: Masa, Fuerza, Aceleración."); return

        if m <= 0:
            messagebox.showwarning("Error",
                "La masa debe ser positiva."); return

        Fnet = fx - ff
        ar   = Fnet / m

        steps = [
            "Datos del problema:",
            f"  m  = {m:.2f} kg",
            f"  F  = {fx:.2f} N",
            f"  Ff = {ff:.2f} N",
            "",
            "Fuerza neta: Fnet = F − Ff",
            f"  Fnet = {fx:.2f} − {ff:.2f}",
            f"  Fnet = {Fnet:.2f} N",
            "",
            "Segunda Ley: a = Fnet / m",
            f"  a = {Fnet:.2f} / {m:.2f}",
            f"  a = {ar:.2f} m/s²",
            "",
            "✔ Verificación: ΣF = m·a",
            f"  {m:.2f} × {ar:.2f} = {m*ar:.2f} N  ✔",
        ]
        self._show_solution(steps, ", ".join(solved) if solved else "")

        datos = segunda_ley(m, fx, ff, tm)
        self._plot_full(datos, "newton",
                        f"Newton  m={m:.2f} kg  F={fx:.2f} N  a={ar:.2f} m/s²")
        self._draw_forces(self.ax_rt, fx, ff)
        self.canvas.draw()
        self._update_stats(v_max=float(np.max(np.abs(datos["v"]))),
                           x_max=float(np.max(datos["x"])),
                           t_total=tm, a_net=ar)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    SimuladorFisica(root)
    root.mainloop()

if __name__ == "__main__":
    main()
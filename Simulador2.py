"""
╔══════════════════════════════════════════════════════╗
║         SIMULADOR DE FÍSICA I  — v2.0               ║
║         Cinemática · Dinámica · Resolución           ║
╚══════════════════════════════════════════════════════╝
Requiere:  pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import math

# ══════════════════════════════════════════════════════
#  PALETA
# ══════════════════════════════════════════════════════
BG       = "#0b0e17"
PANEL    = "#131720"
CARD     = "#1a1f2e"
BORDER   = "#252b3b"
ACCENT   = "#38bdf8"
ACCENT2  = "#f59e0b"
GREEN    = "#34d399"
RED      = "#f87171"
TEXT     = "#f0f4ff"
SUBTEXT  = "#64748b"
MUTED    = "#334155"
INPUT_BG = "#0f1420"

F_MONO = "Consolas"
F_SANS = "Segoe UI"


# ══════════════════════════════════════════════════════
#  FÍSICA
# ══════════════════════════════════════════════════════

def resolver_mrua(v0=None, vf=None, a=None, t=None, x=None):
    known = {k: v for k, v in
             {"v0": v0, "vf": vf, "a": a, "t": t, "x": x}.items()
             if v is not None}
    if len(known) < 3:
        return {"error": "Necesitas al menos 3 valores conocidos."}

    for _ in range(15):
        if "v0" not in known and "vf" in known and "a" in known and "t" in known:
            known["v0"] = known["vf"] - known["a"] * known["t"]
        if "vf" not in known and "v0" in known and "a" in known and "t" in known:
            known["vf"] = known["v0"] + known["a"] * known["t"]
        if "a" not in known and "v0" in known and "vf" in known and "t" in known:
            if known["t"] != 0:
                known["a"] = (known["vf"] - known["v0"]) / known["t"]
        if "t" not in known and "v0" in known and "vf" in known and "a" in known:
            if known["a"] != 0:
                known["t"] = (known["vf"] - known["v0"]) / known["a"]
        if "x" not in known and "v0" in known and "a" in known and "t" in known:
            known["x"] = known["v0"] * known["t"] + 0.5 * known["a"] * known["t"] ** 2
        if "x" not in known and "v0" in known and "vf" in known and "a" in known:
            if known["a"] != 0:
                known["x"] = (known["vf"] ** 2 - known["v0"] ** 2) / (2 * known["a"])
        if "a" not in known and "v0" in known and "vf" in known and "x" in known:
            if known["x"] != 0:
                known["a"] = (known["vf"] ** 2 - known["v0"] ** 2) / (2 * known["x"])
        if "v0" not in known and "x" in known and "a" in known and "t" in known:
            if known["t"] != 0:
                known["v0"] = (known["x"] - 0.5 * known["a"] * known["t"] ** 2) / known["t"]
        if "vf" not in known and "x" in known and "v0" in known and "t" in known:
            if known["t"] != 0:
                known["vf"] = 2 * known["x"] / known["t"] - known["v0"]
        if len(known) == 5:
            break

    if len(known) < 5:
        return {"error": "No se pudo resolver con esa combinación.\nIntenta con otras variables conocidas."}

    t_arr = np.linspace(0, abs(known["t"]), 400)
    x_arr = known["v0"] * t_arr + 0.5 * known["a"] * t_arr ** 2
    v_arr = known["v0"] + known["a"] * t_arr
    return {**known, "t_arr": t_arr, "x_arr": x_arr, "v_arr": v_arr}


def resolver_parabolico(v0, angulo):
    g = 9.8
    rad = math.radians(angulo)
    v0x = v0 * math.cos(rad)
    v0y = v0 * math.sin(rad)
    t_vuelo = 2 * v0y / g
    x_max = v0x * t_vuelo
    y_max = v0y ** 2 / (2 * g)
    t_arr = np.linspace(0, t_vuelo, 500)
    x_arr = v0x * t_arr
    y_arr = v0y * t_arr - 0.5 * g * t_arr ** 2
    vy_arr = v0y - g * t_arr
    v_arr = np.sqrt(v0x ** 2 + vy_arr ** 2)
    return {"v0": v0, "angulo": angulo, "v0x": v0x, "v0y": v0y,
            "t_vuelo": t_vuelo, "x_max": x_max, "y_max": y_max,
            "t_arr": t_arr, "x_arr": x_arr, "y_arr": y_arr, "v_arr": v_arr}


def resolver_newton(masa, fuerzas, rozamiento):
    F_neta = sum(fuerzas) - rozamiento
    a = F_neta / masa
    t_arr = np.linspace(0, 10, 400)
    x_arr = 0.5 * a * t_arr ** 2
    v_arr = a * t_arr
    return {"masa": masa, "F_neta": F_neta, "a": a,
            "t_arr": t_arr, "x_arr": x_arr, "v_arr": v_arr}


def resolver_plano(masa, angulo, uk):
    g = 9.8
    rad = math.radians(angulo)
    N = masa * g * math.cos(rad)
    Fg_par = masa * g * math.sin(rad)
    Ff = uk * N
    F_neta = Fg_par - Ff
    a = F_neta / masa
    t_arr = np.linspace(0, 5, 400)
    x_arr = 0.5 * a * t_arr ** 2
    v_arr = a * t_arr
    return {"masa": masa, "angulo": angulo, "uk": uk,
            "N": N, "Fg_par": Fg_par, "Ff": Ff,
            "F_neta": F_neta, "a": a,
            "t_arr": t_arr, "x_arr": x_arr, "v_arr": v_arr}


# ══════════════════════════════════════════════════════
#  TOOLTIP
# ══════════════════════════════════════════════════════

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tw, text=self.text, bg="#1e293b", fg=ACCENT,
                 font=(F_MONO, 8), relief="flat", bd=0,
                 padx=10, pady=5).pack()

    def hide(self, _=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


# ══════════════════════════════════════════════════════
#  HELPER: CAMPO DE ENTRADA
# ══════════════════════════════════════════════════════

def make_entry(parent, label, tooltip="", default="", hint=""):
    row = tk.Frame(parent, bg=CARD)
    row.pack(fill="x", pady=3)

    lbl = tk.Label(row, text=label, bg=CARD, fg=TEXT,
                   font=(F_SANS, 9), width=24, anchor="w")
    lbl.pack(side="left", padx=(8, 4))
    if tooltip:
        ToolTip(lbl, tooltip)

    var = tk.StringVar(value=default)
    ent = tk.Entry(row, textvariable=var, bg=INPUT_BG, fg=TEXT,
                   font=(F_MONO, 10), width=13, relief="flat",
                   insertbackground=ACCENT, bd=6,
                   highlightthickness=1,
                   highlightbackground=BORDER,
                   highlightcolor=ACCENT)
    ent.pack(side="left", padx=(0, 6))

    if hint:
        tk.Label(row, text=hint, bg=CARD, fg=SUBTEXT,
                 font=(F_SANS, 8)).pack(side="left")

    return var


# ══════════════════════════════════════════════════════
#  PANEL DERECHO: RESULTADOS + GRÁFICA UNIFICADOS
# ══════════════════════════════════════════════════════

class ResultPanel:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=PANEL)

        # Título
        tk.Frame(self.frame, bg=BORDER, height=1).pack(fill="x")
        self.title_var = tk.StringVar(value="Resultados")
        tk.Label(self.frame, textvariable=self.title_var,
                 bg=PANEL, fg=ACCENT,
                 font=(F_MONO, 11, "bold"), pady=10).pack(fill="x", padx=16)

        # Área de texto
        txt_frame = tk.Frame(self.frame, bg=CARD)
        txt_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.text = tk.Text(txt_frame, bg=CARD, fg=GREEN,
                            font=(F_MONO, 9), relief="flat",
                            height=11, bd=10, wrap="word",
                            state="disabled")
        self.text.pack(fill="x")

        # Gráfica
        plt.style.use("dark_background")
        self.fig = plt.Figure(figsize=(5.8, 4.0), facecolor=PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor=CARD)
        ax.text(0.5, 0.5, "Ingresa los datos\ny presiona  ▶  Simular",
                transform=ax.transAxes, ha="center", va="center",
                color=SUBTEXT, fontsize=11, fontfamily=F_MONO)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        self.canvas.draw()

    def update(self, title, texto, plot_fn):
        self.title_var.set(title)
        self.text.configure(state="normal", fg=GREEN)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", texto)
        self.text.configure(state="disabled")
        self.fig.clear()
        plot_fn(self.fig)
        self.canvas.draw()

    def error(self, msg):
        self.title_var.set("⚠  Error")
        self.text.configure(state="normal", fg=RED)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n  " + msg)
        self.text.configure(state="disabled")
        self._placeholder()


# ══════════════════════════════════════════════════════
#  ESTILOS DE EJES
# ══════════════════════════════════════════════════════

def ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=9, fontfamily=F_MONO, pad=6)
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8)
    ax.tick_params(colors=SUBTEXT, labelsize=7)
    ax.grid(alpha=0.1, color=SUBTEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# ══════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════

def build_mrua(parent, rp: ResultPanel):
    tab = tk.Frame(parent, bg=BG)

    # Cabecera
    hdr = tk.Frame(tab, bg=CARD, pady=10, padx=14)
    hdr.pack(fill="x", padx=12, pady=(12, 4))
    tk.Label(hdr, text="MRUA  —  Movimiento Rectilíneo Uniformemente Acelerado",
             bg=CARD, fg=ACCENT, font=(F_MONO, 9, "bold")).pack(anchor="w")
    tk.Label(hdr,
             text="Deja VACÍO la variable que quieres calcular (incógnita).",
             bg=CARD, fg=ACCENT2, font=(F_SANS, 8, "italic")).pack(anchor="w", pady=(3, 0))

    # Ecuaciones
    eq = tk.Frame(tab, bg=PANEL, pady=5)
    eq.pack(fill="x", padx=12, pady=(0, 6))
    for e in ["  vf = v₀ + a·t",
              "  x  = v₀·t + ½·a·t²",
              "  vf² = v₀² + 2·a·x"]:
        tk.Label(eq, text=e, bg=PANEL, fg=ACCENT2,
                 font=(F_MONO, 8)).pack(anchor="w")

    # Entradas
    inp = tk.Frame(tab, bg=CARD, padx=10, pady=10)
    inp.pack(fill="x", padx=12)
    tk.Label(inp, text="  Variables  (vacío = incógnita a resolver)",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8, "italic")).pack(anchor="w", pady=(0, 5))

    v0 = make_entry(inp, "v₀  Veloc. inicial", "Velocidad al inicio (m/s)", "0", "m/s")
    vf = make_entry(inp, "vf  Veloc. final", "Velocidad al final (m/s)", "", "m/s  ← incógnita?")
    a  = make_entry(inp, "a   Aceleración", "Cambio de velocidad por segundo (m/s²)", "2", "m/s²")
    t  = make_entry(inp, "t   Tiempo", "Duración del movimiento (s)", "5", "s")
    x  = make_entry(inp, "x   Desplazamiento", "Distancia recorrida (m)", "", "m  ← incógnita?")

    def _get(var):
        s = var.get().strip()
        return float(s) if s else None

    def simular():
        try:
            res = resolver_mrua(_get(v0), _get(vf), _get(a), _get(t), _get(x))
            if "error" in res:
                rp.error(res["error"]); return

            texto = (
                f"  Velocidad inicial  v₀ = {res['v0']:>12.4f}  m/s\n"
                f"  Velocidad final    vf = {res['vf']:>12.4f}  m/s\n"
                f"  Aceleración         a = {res['a']:>12.4f}  m/s²\n"
                f"  Tiempo              t = {res['t']:>12.4f}  s\n"
                f"  Desplazamiento      x = {res['x']:>12.4f}  m\n"
                f"\n"
                f"  ── Verificación ─────────────────────\n"
                f"  vf = v₀ + a·t\n"
                f"     = {res['v0']:.3f} + {res['a']:.3f}·{res['t']:.3f}\n"
                f"     = {res['vf']:.4f} m/s  ✔\n"
                f"  x  = v₀·t + ½·a·t²\n"
                f"     = {res['x']:.4f} m  ✔\n"
            )

            def plot(fig):
                gs = GridSpec(1, 2, figure=fig, wspace=0.35)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax_style(ax1, "Posición vs Tiempo", "t (s)", "x (m)")
                ax_style(ax2, "Velocidad vs Tiempo", "t (s)", "v (m/s)")
                ax1.plot(res["t_arr"], res["x_arr"], color=ACCENT, lw=2)
                ax1.fill_between(res["t_arr"], 0, res["x_arr"], alpha=0.07, color=ACCENT)
                ax2.plot(res["t_arr"], res["v_arr"], color=ACCENT2, lw=2)
                ax2.axhline(0, color=MUTED, lw=0.8, ls="--")
                fig.tight_layout(pad=1.6)

            rp.update("MRUA — Resultados", texto, plot)
        except ValueError:
            rp.error("Valor inválido. Usa números (ej: 9.8,  -2,  0).")

    # Botones
    br = tk.Frame(tab, bg=BG)
    br.pack(fill="x", padx=12, pady=10)
    tk.Button(br, text="▶  Simular / Resolver",
              bg=ACCENT, fg=BG, font=(F_MONO, 10, "bold"),
              relief="flat", bd=0, padx=18, pady=9,
              cursor="hand2", command=simular).pack(side="left")
    tk.Button(br, text="↺  Limpiar",
              bg=CARD, fg=SUBTEXT, font=(F_MONO, 9),
              relief="flat", bd=0, padx=12, pady=9, cursor="hand2",
              command=lambda: [v.set("") for v in [v0, vf, a, t, x]]
              ).pack(side="left", padx=8)
    return tab


def build_parabolico(parent, rp: ResultPanel):
    tab = tk.Frame(parent, bg=BG)

    hdr = tk.Frame(tab, bg=CARD, pady=10, padx=14)
    hdr.pack(fill="x", padx=12, pady=(12, 4))
    tk.Label(hdr, text="Tiro Parabólico  —  Lanzamiento con ángulo",
             bg=CARD, fg=ACCENT, font=(F_MONO, 9, "bold")).pack(anchor="w")
    tk.Label(hdr, text="Movimiento bidimensional bajo la gravedad  g = 9.8 m/s²",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8)).pack(anchor="w", pady=(3, 0))

    eq = tk.Frame(tab, bg=PANEL, pady=5)
    eq.pack(fill="x", padx=12, pady=(0, 6))
    for e in ["  x(t) = v₀·cos(θ)·t",
              "  y(t) = v₀·sin(θ)·t − ½·g·t²",
              "  R    = v₀²·sin(2θ) / g"]:
        tk.Label(eq, text=e, bg=PANEL, fg=ACCENT2, font=(F_MONO, 8)).pack(anchor="w")

    inp = tk.Frame(tab, bg=CARD, padx=10, pady=10)
    inp.pack(fill="x", padx=12)
    v0  = make_entry(inp, "v₀  Veloc. inicial", "Velocidad de lanzamiento (m/s)", "20", "m/s")
    ang = make_entry(inp, "θ   Ángulo", "Ángulo respecto a la horizontal (0°–90°)", "45", "°")
    tk.Label(inp, text="  → Calcula automáticamente: alcance · altura máx · tiempo vuelo",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8, "italic")).pack(anchor="w", pady=(6, 0))

    def simular():
        try:
            v0_v  = float(v0.get())
            ang_v = float(ang.get())
            if not (0 < ang_v < 90):
                rp.error("El ángulo debe estar entre 0° y 90°."); return
            res = resolver_parabolico(v0_v, ang_v)
            texto = (
                f"  Velocidad inicial    v₀ = {res['v0']:.4f}  m/s\n"
                f"  Ángulo               θ  = {res['angulo']:.2f}  °\n"
                f"  ── Componentes ──────────────────────\n"
                f"  Horizontal          v₀ₓ = {res['v0x']:.4f}  m/s\n"
                f"  Vertical            v₀ᵧ = {res['v0y']:.4f}  m/s\n"
                f"  ── Resultados ───────────────────────\n"
                f"  Tiempo de vuelo       t = {res['t_vuelo']:.4f}  s\n"
                f"  Alcance horizontal    R = {res['x_max']:.4f}  m\n"
                f"  Altura máxima      ymax = {res['y_max']:.4f}  m\n"
            )

            def plot(fig):
                gs = GridSpec(1, 2, figure=fig, wspace=0.35)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax_style(ax1, "Trayectoria", "x (m)", "y (m)")
                ax_style(ax2, "Velocidad vs Tiempo", "t (s)", "|v| (m/s)")
                ax1.plot(res["x_arr"], res["y_arr"], color=ACCENT, lw=2)
                ax1.fill_between(res["x_arr"], 0, res["y_arr"], alpha=0.07, color=ACCENT)
                ax1.axhline(0, color=MUTED, lw=0.8)
                ax1.plot(res["x_max"] / 2, res["y_max"], "o", color=ACCENT2, ms=6,
                         label=f"ymax={res['y_max']:.1f}m")
                ax1.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
                ax2.plot(res["t_arr"], res["v_arr"], color=ACCENT2, lw=2)
                fig.tight_layout(pad=1.6)

            rp.update("Tiro Parabólico — Resultados", texto, plot)
        except ValueError:
            rp.error("Ingresa valores numéricos válidos.")

    br = tk.Frame(tab, bg=BG)
    br.pack(fill="x", padx=12, pady=10)
    tk.Button(br, text="▶  Simular",
              bg=ACCENT, fg=BG, font=(F_MONO, 10, "bold"),
              relief="flat", bd=0, padx=18, pady=9,
              cursor="hand2", command=simular).pack(side="left")
    return tab


def build_newton(parent, rp: ResultPanel):
    tab = tk.Frame(parent, bg=BG)

    hdr = tk.Frame(tab, bg=CARD, pady=10, padx=14)
    hdr.pack(fill="x", padx=12, pady=(12, 4))
    tk.Label(hdr, text="2ª Ley de Newton  —  ΣF = m · a",
             bg=CARD, fg=ACCENT, font=(F_MONO, 9, "bold")).pack(anchor="w")
    tk.Label(hdr, text="Suma de fuerzas sobre un cuerpo — movimiento en 1 dimensión",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8)).pack(anchor="w", pady=(3, 0))

    inp = tk.Frame(tab, bg=CARD, padx=10, pady=10)
    inp.pack(fill="x", padx=12)
    masa = make_entry(inp, "m   Masa", "Masa del cuerpo en kg", "10", "kg")
    tk.Label(inp, text="  Fuerzas en la dirección del movimiento:",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8, "italic")).pack(anchor="w", pady=(8, 4))
    f1 = make_entry(inp, "F₁  Primera fuerza", "Negativa = dirección opuesta", "30", "N")
    f2 = make_entry(inp, "F₂  Segunda fuerza", "0 si no hay segunda fuerza", "0", "N  (opcional)")
    ff = make_entry(inp, "Ff  Rozamiento", "Fricción — siempre opuesta al movimiento", "5", "N")

    def simular():
        try:
            m    = float(masa.get())
            f1_v = float(f1.get() or 0)
            f2_v = float(f2.get() or 0)
            ff_v = float(ff.get() or 0)
            if m <= 0:
                rp.error("La masa debe ser mayor que 0."); return
            res = resolver_newton(m, [f1_v, f2_v], ff_v)
            texto = (
                f"  Masa                   m = {res['masa']:.4f}  kg\n"
                f"  Fuerzas               F₁ = {f1_v:.2f}  N\n"
                f"                        F₂ = {f2_v:.2f}  N\n"
                f"  Rozamiento            Ff = {ff_v:.4f}  N\n"
                f"  ── Resultados ───────────────────────\n"
                f"  Fuerza neta          ΣF  = {res['F_neta']:.4f}  N\n"
                f"  Aceleración           a  = {res['a']:.4f}  m/s²\n"
                f"\n"
                f"  ── Cálculo ──────────────────────────\n"
                f"  ΣF = F₁ + F₂ − Ff\n"
                f"     = {f1_v} + {f2_v} − {ff_v}\n"
                f"     = {res['F_neta']:.4f} N\n"
                f"  a  = ΣF / m = {res['F_neta']:.4f} / {m}\n"
                f"     = {res['a']:.4f} m/s²\n"
            )

            def plot(fig):
                gs = GridSpec(1, 2, figure=fig, wspace=0.35)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax_style(ax1, "Posición vs Tiempo", "t (s)", "x (m)")
                ax_style(ax2, "Velocidad vs Tiempo", "t (s)", "v (m/s)")
                ax1.plot(res["t_arr"], res["x_arr"], color=ACCENT, lw=2)
                ax1.fill_between(res["t_arr"], 0, res["x_arr"], alpha=0.07, color=ACCENT)
                ax2.plot(res["t_arr"], res["v_arr"], color=ACCENT2, lw=2)
                ax2.axhline(0, color=MUTED, lw=0.8, ls="--")
                fig.tight_layout(pad=1.6)

            rp.update("2ª Ley Newton — Resultados", texto, plot)
        except ValueError:
            rp.error("Ingresa valores numéricos válidos.")

    br = tk.Frame(tab, bg=BG)
    br.pack(fill="x", padx=12, pady=10)
    tk.Button(br, text="▶  Simular",
              bg=ACCENT, fg=BG, font=(F_MONO, 10, "bold"),
              relief="flat", bd=0, padx=18, pady=9,
              cursor="hand2", command=simular).pack(side="left")
    return tab


def build_plano(parent, rp: ResultPanel):
    tab = tk.Frame(parent, bg=BG)

    hdr = tk.Frame(tab, bg=CARD, pady=10, padx=14)
    hdr.pack(fill="x", padx=12, pady=(12, 4))
    tk.Label(hdr, text="Plano Inclinado  —  Fuerzas y Fricción",
             bg=CARD, fg=ACCENT, font=(F_MONO, 9, "bold")).pack(anchor="w")
    tk.Label(hdr, text="Bloque sobre superficie inclinada con o sin rozamiento",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8)).pack(anchor="w", pady=(3, 0))

    eq = tk.Frame(tab, bg=PANEL, pady=5)
    eq.pack(fill="x", padx=12, pady=(0, 6))
    for e in ["  N  = m·g·cos(θ)",
              "  Ff = μₖ · N",
              "  a  = g·(sin θ − μₖ·cos θ)"]:
        tk.Label(eq, text=e, bg=PANEL, fg=ACCENT2, font=(F_MONO, 8)).pack(anchor="w")

    inp = tk.Frame(tab, bg=CARD, padx=10, pady=10)
    inp.pack(fill="x", padx=12)
    masa = make_entry(inp, "m   Masa del bloque", "Masa en kilogramos", "8", "kg")
    ang  = make_entry(inp, "θ   Ángulo del plano", "Ángulo de inclinación (1°–89°)", "30", "°")
    uk   = make_entry(inp, "μₖ  Coef. rozamiento", "0 = superficie sin fricción", "0.2", "adim.")

    def simular():
        try:
            m_v  = float(masa.get())
            a_v  = float(ang.get())
            uk_v = float(uk.get())
            if not (0 < a_v < 90):
                rp.error("El ángulo debe estar entre 1° y 89°."); return
            if uk_v < 0:
                rp.error("El coef. de rozamiento no puede ser negativo."); return
            res = resolver_plano(m_v, a_v, uk_v)
            if res["a"] > 0.001:
                estado = "▼ Desliza hacia abajo"
            elif res["a"] < -0.001:
                estado = "▲ Asciende"
            else:
                estado = "■ En reposo (equilibrio)"

            texto = (
                f"  ── Análisis de fuerzas ──────────────\n"
                f"  Normal               N  = {res['N']:.4f}  N\n"
                f"  Peso paralelo     Fg∥   = {res['Fg_par']:.4f}  N\n"
                f"  Fricción cinética  Ff   = {res['Ff']:.4f}  N\n"
                f"  Fuerza neta        ΣF   = {res['F_neta']:.4f}  N\n"
                f"  ── Movimiento ───────────────────────\n"
                f"  Aceleración          a  = {res['a']:.4f}  m/s²\n"
                f"  Estado del bloque:  {estado}\n"
                f"\n"
                f"  ── Cálculo ──────────────────────────\n"
                f"  N   = {m_v}·9.8·cos({a_v}°) = {res['N']:.4f} N\n"
                f"  Ff  = {uk_v}·{res['N']:.4f}  = {res['Ff']:.4f} N\n"
                f"  ΣF  = {res['Fg_par']:.3f} − {res['Ff']:.3f}\n"
                f"      = {res['F_neta']:.4f} N\n"
            )

            def plot(fig):
                gs = GridSpec(1, 2, figure=fig, wspace=0.35)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax_style(ax1, "Desplazamiento vs t", "t (s)", "x (m)")
                ax_style(ax2, "Velocidad vs t", "t (s)", "v (m/s)")
                mask = res["x_arr"] >= 0
                ax1.plot(res["t_arr"][mask], res["x_arr"][mask], color=ACCENT, lw=2)
                ax1.fill_between(res["t_arr"][mask], 0, res["x_arr"][mask],
                                 alpha=0.07, color=ACCENT)
                ax2.plot(res["t_arr"][mask], res["v_arr"][mask], color=ACCENT2, lw=2)
                ax2.axhline(0, color=MUTED, lw=0.8, ls="--")
                fig.tight_layout(pad=1.6)

            rp.update("Plano Inclinado — Resultados", texto, plot)
        except ValueError:
            rp.error("Ingresa valores numéricos válidos.")

    br = tk.Frame(tab, bg=BG)
    br.pack(fill="x", padx=12, pady=10)
    tk.Button(br, text="▶  Simular",
              bg=ACCENT, fg=BG, font=(F_MONO, 10, "bold"),
              relief="flat", bd=0, padx=18, pady=9,
              cursor="hand2", command=simular).pack(side="left")
    return tab


# ══════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("Simulador de Física I  —  v2.0")
    root.configure(bg=BG)
    root.geometry("1120x720")
    root.minsize(900, 620)

    # Header
    hdr = tk.Frame(root, bg="#0d1120")
    hdr.pack(fill="x")
    inner = tk.Frame(hdr, bg="#0d1120")
    inner.pack(fill="x", padx=24, pady=14)
    tk.Label(inner, text="⚛  FÍSICA I",
             bg="#0d1120", fg=ACCENT,
             font=(F_MONO, 16, "bold")).pack(side="left")
    tk.Label(inner, text="  Simulador  —  Cinemática & Dinámica",
             bg="#0d1120", fg=TEXT,
             font=(F_MONO, 12)).pack(side="left")
    tk.Label(inner, text=" v2.0 ",
             bg=ACCENT, fg=BG,
             font=(F_MONO, 8, "bold"),
             padx=4, pady=2).pack(side="right")
    tk.Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # Layout: izquierda (controles) | derecha (resultados + gráfica)
    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True)

    left = tk.Frame(body, bg=BG, width=440)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)

    right = tk.Frame(body, bg=PANEL)
    right.pack(side="left", fill="both", expand=True)

    rp = ResultPanel(right)
    rp.frame.pack(fill="both", expand=True)

    # Tabs
    style = ttk.Style()
    style.theme_use("default")
    style.configure("P.TNotebook",
                    background=BG, borderwidth=0, tabmargins=[0, 6, 0, 0])
    style.configure("P.TNotebook.Tab",
                    background=CARD, foreground=SUBTEXT,
                    font=(F_MONO, 9), padding=[14, 8], borderwidth=0)
    style.map("P.TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", BG)],
              font=[("selected", (F_MONO, 9, "bold"))])

    nb = ttk.Notebook(left, style="P.TNotebook")
    nb.pack(fill="both", expand=True)

    nb.add(build_mrua(nb, rp),        text="  MRUA  ")
    nb.add(build_parabolico(nb, rp),  text="  Parabólico  ")
    nb.add(build_newton(nb, rp),      text="  2ª Newton  ")
    nb.add(build_plano(nb, rp),       text="  Plano Incl.  ")

    # Status bar
    sb = tk.Frame(root, bg=CARD, pady=5)
    sb.pack(fill="x", side="bottom")
    tk.Label(sb,
             text="  💡  MRUA: deja vacía la incógnita → el simulador la calcula automáticamente  "
                  "·  Pasa el cursor sobre las etiquetas para ver ayuda",
             bg=CARD, fg=SUBTEXT, font=(F_SANS, 8)).pack(side="left")

    root.mainloop()


if __name__ == "__main__":
    main()
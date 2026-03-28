"""
Simulador de Física 1 - Cinemática y Dinámica
Universidad - Física I
Requiere: pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ─────────────────────────────────────────────
#  ESTILOS / CONSTANTES
# ─────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
ACCENT   = "#4f8ef7"
ACCENT2  = "#f7a14f"
TEXT     = "#e8eaf0"
SUBTEXT  = "#8890a4"
SUCCESS  = "#4fca7a"
BORDER   = "#2a2d3a"

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_SUB    = ("Courier New", 11, "bold")
FONT_LABEL  = ("Courier New", 10)
FONT_SMALL  = ("Courier New", 9)
FONT_RESULT = ("Courier New", 10, "bold")

# ─────────────────────────────────────────────
#  FÍSICA: CINEMÁTICA
# ─────────────────────────────────────────────

def cinematica_mrua(v0, a, t_max, angulo=0):
    """Movimiento rectilíneo uniformemente acelerado (puede ser proyectil)"""
    ang_rad = math.radians(angulo)
    v0x = v0 * math.cos(ang_rad)
    v0y = v0 * math.sin(ang_rad)
    g   = 9.8

    t = np.linspace(0, t_max, 500)

    if angulo == 0:
        # MRUA puro
        x  = v0x * t + 0.5 * a * t**2
        vx = v0x + a * t
        ax = np.full_like(t, a)
        return {"t": t, "x": x, "v": vx, "a": ax, "tipo": "MRUA"}
    else:
        # Tiro parabólico
        x  = v0x * t
        y  = v0y * t - 0.5 * g * t**2
        vx = np.full_like(t, v0x)
        vy = v0y - g * t
        v  = np.sqrt(vx**2 + vy**2)
        # Truncar cuando y < 0
        idx = np.where(y < 0)[0]
        if len(idx) > 0:
            cut = idx[0]
            x, y, t, v = x[:cut], y[:cut], t[:cut], v[:cut]
        return {"t": t, "x": x, "y": y, "v": v, "tipo": "Parabólico",
                "x_max": x[-1] if len(x) > 0 else 0,
                "y_max": float(np.max(y)) if len(y) > 0 else 0}


# ─────────────────────────────────────────────
#  FÍSICA: DINÁMICA
# ─────────────────────────────────────────────

def segunda_ley(masa, fuerzas_x, fuerzas_y, rozamiento=0, t_max=10):
    """Segunda Ley de Newton con múltiples fuerzas"""
    Fx = sum(fuerzas_x) - rozamiento
    Fy = sum(fuerzas_y)
    ax = Fx / masa
    ay = Fy / masa
    a_total = math.sqrt(ax**2 + ay**2)

    t  = np.linspace(0, t_max, 500)
    vx = ax * t
    vy = ay * t
    v  = np.sqrt(vx**2 + vy**2)
    x  = 0.5 * ax * t**2
    y  = 0.5 * ay * t**2

    return {"t": t, "ax": ax, "ay": ay, "a": a_total,
            "vx": vx, "vy": vy, "v": v, "x": x, "y": y,
            "Fx": Fx, "Fy": Fy}


def plano_inclinado(masa, angulo, uk=0.0):
    """Bloque en plano inclinado con fricción"""
    g       = 9.8
    ang_rad = math.radians(angulo)
    N       = masa * g * math.cos(ang_rad)
    Fg_par  = masa * g * math.sin(ang_rad)
    Ff      = uk * N
    F_neta  = Fg_par - Ff
    a       = F_neta / masa

    t_max = 5 if a > 0 else 2
    t     = np.linspace(0, t_max, 500)
    x     = 0.5 * a * t**2
    v     = a * t

    return {"t": t, "x": x, "v": v, "a": a,
            "N": N, "Fg_par": Fg_par, "Ff": Ff, "F_neta": F_neta}


# ─────────────────────────────────────────────
#  VENTANA DE GRÁFICAS
# ─────────────────────────────────────────────

def mostrar_grafica(datos, titulo, modo):
    win = tk.Toplevel()
    win.title(titulo)
    win.configure(bg=BG)
    win.geometry("900x600")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             facecolor="#0f1117")
    fig.suptitle(titulo, color=TEXT, fontsize=13,
                 fontfamily="Courier New", fontweight="bold")

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    t = datos["t"]

    if modo == "mrua":
        axes[0].plot(t, datos["x"], color=ACCENT, lw=2)
        axes[0].set_title("Posición vs Tiempo", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[0].set_xlabel("t (s)", color=SUBTEXT, fontsize=8)
        axes[0].set_ylabel("x (m)", color=SUBTEXT, fontsize=8)
        axes[0].grid(alpha=0.15)

        axes[1].plot(t, datos["v"], color=ACCENT2, lw=2)
        axes[1].set_title("Velocidad vs Tiempo", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=8)
        axes[1].set_ylabel("v (m/s)", color=SUBTEXT, fontsize=8)
        axes[1].grid(alpha=0.15)

    elif modo == "parabolico":
        axes[0].plot(datos["x"], datos["y"], color=ACCENT, lw=2)
        axes[0].fill_between(datos["x"], 0, datos["y"],
                             alpha=0.08, color=ACCENT)
        axes[0].set_title("Trayectoria", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[0].set_xlabel("x (m)", color=SUBTEXT, fontsize=8)
        axes[0].set_ylabel("y (m)", color=SUBTEXT, fontsize=8)
        axes[0].grid(alpha=0.15)

        axes[1].plot(t, datos["v"], color=ACCENT2, lw=2)
        axes[1].set_title("Velocidad vs Tiempo", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=8)
        axes[1].set_ylabel("|v| (m/s)", color=SUBTEXT, fontsize=8)
        axes[1].grid(alpha=0.15)

    elif modo in ("newton", "plano"):
        axes[0].plot(t, datos["x"], color=ACCENT, lw=2)
        axes[0].set_title("Posición vs Tiempo", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[0].set_xlabel("t (s)", color=SUBTEXT, fontsize=8)
        axes[0].set_ylabel("x (m)", color=SUBTEXT, fontsize=8)
        axes[0].grid(alpha=0.15)

        axes[1].plot(t, datos["v"], color=ACCENT2, lw=2)
        axes[1].set_title("Velocidad vs Tiempo", color=TEXT,
                          fontsize=10, fontfamily="Courier New")
        axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=8)
        axes[1].set_ylabel("v (m/s)", color=SUBTEXT, fontsize=8)
        axes[1].grid(alpha=0.15)

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True,
                                padx=10, pady=10)

    tk.Button(win, text="✕  Cerrar", bg=BORDER, fg=TEXT,
              font=FONT_SMALL, relief="flat", bd=0,
              padx=12, pady=6,
              command=win.destroy).pack(pady=(0, 10))


# ─────────────────────────────────────────────
#  HELPERS UI
# ─────────────────────────────────────────────

def make_entry(parent, label, default="0", width=14):
    frame = tk.Frame(parent, bg=PANEL)
    frame.pack(fill="x", pady=3)
    tk.Label(frame, text=label, bg=PANEL, fg=SUBTEXT,
             font=FONT_LABEL, width=28, anchor="w").pack(side="left")
    var = tk.StringVar(value=default)
    ent = tk.Entry(frame, textvariable=var, bg=BORDER,
                   fg=TEXT, font=FONT_LABEL, width=width,
                   relief="flat", insertbackground=TEXT,
                   bd=4)
    ent.pack(side="left", padx=(4, 0))
    return var


def resultado_box(parent, texto):
    """Muestra un cuadro de resultados"""
    win = tk.Toplevel(parent)
    win.title("Resultados")
    win.configure(bg=BG)
    win.geometry("460x360")
    win.resizable(False, False)

    tk.Label(win, text="RESULTADOS", bg=BG, fg=ACCENT,
             font=FONT_SUB).pack(pady=(18, 4))

    box = tk.Text(win, bg=PANEL, fg=SUCCESS,
                  font=FONT_RESULT, relief="flat", bd=8,
                  wrap="word", height=14)
    box.insert("1.0", texto)
    box.configure(state="disabled")
    box.pack(padx=20, pady=8, fill="both", expand=True)

    tk.Button(win, text="Cerrar", bg=ACCENT, fg=BG,
              font=FONT_SUB, relief="flat", bd=0,
              padx=16, pady=6,
              command=win.destroy).pack(pady=(0, 14))


# ─────────────────────────────────────────────
#  PANEL: CINEMÁTICA MRUA
# ─────────────────────────────────────────────

def tab_mrua(notebook):
    tab = tk.Frame(notebook, bg=BG)
    notebook.add(tab, text="  MRUA  ")

    tk.Label(tab, text="Movimiento Rectilíneo\nUniformemente Acelerado",
             bg=BG, fg=ACCENT, font=FONT_SUB,
             justify="center").pack(pady=(20, 4))
    tk.Label(tab, text="x = x₀ + v₀·t + ½·a·t²",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(pady=(0, 14))

    frame = tk.Frame(tab, bg=PANEL, bd=0, relief="flat",
                     padx=24, pady=18)
    frame.pack(padx=40, fill="x")

    v0  = make_entry(frame, "Velocidad inicial v₀  (m/s)", "10")
    a   = make_entry(frame, "Aceleración a  (m/s²)", "2")
    t   = make_entry(frame, "Tiempo máximo  (s)", "5")

    def calcular():
        try:
            datos = cinematica_mrua(float(v0.get()), float(a.get()),
                                    float(t.get()))
            x_f = datos["x"][-1]
            v_f = datos["v"][-1]
            res = (
                f"► Posición final:        {x_f:.3f} m\n"
                f"► Velocidad final:       {v_f:.3f} m/s\n"
                f"► Aceleración:           {float(a.get()):.3f} m/s²\n\n"
                f"  Ecuaciones usadas:\n"
                f"  x(t) = {float(v0.get())}·t + ½·({float(a.get())})·t²\n"
                f"  v(t) = {float(v0.get())} + {float(a.get())}·t\n"
            )
            resultado_box(tab, res)
            mostrar_grafica(datos, "MRUA – Cinemática", "mrua")
        except ValueError:
            messagebox.showerror("Error", "Ingresa valores numéricos válidos.")

    tk.Button(tab, text="▶  Simular", bg=ACCENT, fg=BG,
              font=FONT_SUB, relief="flat", bd=0,
              padx=20, pady=8, cursor="hand2",
              command=calcular).pack(pady=22)
    return tab


# ─────────────────────────────────────────────
#  PANEL: TIRO PARABÓLICO
# ─────────────────────────────────────────────

def tab_parabolico(notebook):
    tab = tk.Frame(notebook, bg=BG)
    notebook.add(tab, text="  Tiro Parabólico  ")

    tk.Label(tab, text="Tiro Parabólico",
             bg=BG, fg=ACCENT, font=FONT_SUB).pack(pady=(20, 4))
    tk.Label(tab, text="y = v₀·sin(θ)·t − ½·g·t²",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(pady=(0, 14))

    frame = tk.Frame(tab, bg=PANEL, bd=0, padx=24, pady=18)
    frame.pack(padx=40, fill="x")

    v0  = make_entry(frame, "Velocidad inicial v₀  (m/s)", "20")
    ang = make_entry(frame, "Ángulo de lanzamiento θ  (°)", "45")
    t   = make_entry(frame, "Tiempo máximo simulación (s)", "4")

    def calcular():
        try:
            a_val = float(ang.get())
            if not (0 < a_val < 90):
                messagebox.showwarning("Advertencia",
                    "El ángulo debe estar entre 0° y 90°.")
                return
            datos = cinematica_mrua(float(v0.get()), 0,
                                    float(t.get()), angulo=a_val)
            res = (
                f"► Alcance horizontal:    {datos['x_max']:.3f} m\n"
                f"► Altura máxima:         {datos['y_max']:.3f} m\n"
                f"► Ángulo de lanzamiento: {a_val}°\n"
                f"► Velocidad inicial:     {float(v0.get())} m/s\n\n"
                f"  Componentes:\n"
                f"  v₀ₓ = {float(v0.get()):.2f}·cos({a_val}°)"
                f" = {float(v0.get())*math.cos(math.radians(a_val)):.3f} m/s\n"
                f"  v₀ᵧ = {float(v0.get()):.2f}·sin({a_val}°)"
                f" = {float(v0.get())*math.sin(math.radians(a_val)):.3f} m/s\n"
            )
            resultado_box(tab, res)
            mostrar_grafica(datos, "Tiro Parabólico", "parabolico")
        except ValueError:
            messagebox.showerror("Error", "Ingresa valores numéricos válidos.")

    tk.Button(tab, text="▶  Simular", bg=ACCENT, fg=BG,
              font=FONT_SUB, relief="flat", bd=0,
              padx=20, pady=8, cursor="hand2",
              command=calcular).pack(pady=22)
    return tab


# ─────────────────────────────────────────────
#  PANEL: 2ª LEY DE NEWTON
# ─────────────────────────────────────────────

def tab_newton(notebook):
    tab = tk.Frame(notebook, bg=BG)
    notebook.add(tab, text="  2ª Ley Newton  ")

    tk.Label(tab, text="Segunda Ley de Newton",
             bg=BG, fg=ACCENT, font=FONT_SUB).pack(pady=(20, 4))
    tk.Label(tab, text="ΣF = m · a",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(pady=(0, 14))

    frame = tk.Frame(tab, bg=PANEL, bd=0, padx=24, pady=18)
    frame.pack(padx=40, fill="x")

    masa  = make_entry(frame, "Masa m  (kg)", "5")
    fx    = make_entry(frame, "Fuerza aplicada Fₓ  (N)", "20")
    ff    = make_entry(frame, "Fuerza de rozamiento Ff  (N)", "3")
    t_max = make_entry(frame, "Tiempo de simulación  (s)", "8")

    def calcular():
        try:
            m   = float(masa.get())
            fxv = float(fx.get())
            ffv = float(ff.get())
            tm  = float(t_max.get())
            if m <= 0:
                messagebox.showwarning("Error", "La masa debe ser positiva.")
                return
            datos = segunda_ley(m, [fxv], [0], rozamiento=ffv, t_max=tm)
            datos["v"] = datos["v"]   # alias
            res = (
                f"► Aceleración resultante: {datos['a']:.4f} m/s²\n"
                f"► Fuerza neta Fx:         {datos['Fx']:.3f} N\n"
                f"► Fuerza neta Fy:         {datos['Fy']:.3f} N\n\n"
                f"  Verificación:\n"
                f"  a = ΣFₓ/m = ({fxv} − {ffv}) / {m}\n"
                f"    = {datos['ax']:.4f} m/s²\n"
            )
            resultado_box(tab, res)
            mostrar_grafica({"t": datos["t"], "x": datos["x"],
                             "v": datos["vx"]},
                            "2ª Ley de Newton", "newton")
        except ValueError:
            messagebox.showerror("Error", "Ingresa valores numéricos válidos.")

    tk.Button(tab, text="▶  Simular", bg=ACCENT, fg=BG,
              font=FONT_SUB, relief="flat", bd=0,
              padx=20, pady=8, cursor="hand2",
              command=calcular).pack(pady=22)
    return tab


# ─────────────────────────────────────────────
#  PANEL: PLANO INCLINADO
# ─────────────────────────────────────────────

def tab_plano(notebook):
    tab = tk.Frame(notebook, bg=BG)
    notebook.add(tab, text="  Plano Inclinado  ")

    tk.Label(tab, text="Bloque en Plano Inclinado",
             bg=BG, fg=ACCENT, font=FONT_SUB).pack(pady=(20, 4))
    tk.Label(tab, text="a = g·(sin θ − μₖ·cos θ)",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(pady=(0, 14))

    frame = tk.Frame(tab, bg=PANEL, bd=0, padx=24, pady=18)
    frame.pack(padx=40, fill="x")

    masa = make_entry(frame, "Masa del bloque m  (kg)", "10")
    ang  = make_entry(frame, "Ángulo del plano θ  (°)", "30")
    uk   = make_entry(frame, "Coef. rozamiento cinético μₖ", "0.2")

    def calcular():
        try:
            m    = float(masa.get())
            a_v  = float(ang.get())
            uk_v = float(uk.get())
            if not (0 < a_v < 90):
                messagebox.showwarning("Advertencia",
                    "El ángulo debe estar entre 0° y 90°.")
                return
            datos = plano_inclinado(m, a_v, uk_v)
            estado = "Desliza ↓" if datos["a"] > 0 else "En reposo / sube"
            res = (
                f"► Estado del bloque:      {estado}\n"
                f"► Aceleración:            {datos['a']:.4f} m/s²\n"
                f"► Normal N:               {datos['N']:.3f} N\n"
                f"► Componente paralela Fg: {datos['Fg_par']:.3f} N\n"
                f"► Fricción Ff:            {datos['Ff']:.3f} N\n"
                f"► Fuerza neta:            {datos['F_neta']:.3f} N\n\n"
                f"  g = 9.8 m/s²,  θ = {a_v}°,  μₖ = {uk_v}\n"
                f"  N = m·g·cos(θ) = {datos['N']:.3f} N\n"
            )
            resultado_box(tab, res)
            mostrar_grafica({"t": datos["t"], "x": datos["x"],
                             "v": datos["v"]},
                            "Plano Inclinado", "plano")
        except ValueError:
            messagebox.showerror("Error", "Ingresa valores numéricos válidos.")

    tk.Button(tab, text="▶  Simular", bg=ACCENT, fg=BG,
              font=FONT_SUB, relief="flat", bd=0,
              padx=20, pady=8, cursor="hand2",
              command=calcular).pack(pady=22)
    return tab


# ─────────────────────────────────────────────
#  VENTANA PRINCIPAL
# ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Simulador de Física I")
    root.configure(bg=BG)
    root.geometry("620x580")
    root.resizable(False, False)

    # ── Header ──────────────────────────────
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=30, pady=(28, 0))

    tk.Label(header, text="FÍSICA I", bg=BG, fg=ACCENT,
             font=FONT_TITLE).pack(side="left")
    tk.Label(header, text=" Simulador",
             bg=BG, fg=TEXT,
             font=("Courier New", 22)).pack(side="left")

    tk.Label(root,
             text="Cinemática  ·  Dinámica de Fuerzas",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w",
                                                      padx=30,
                                                      pady=(2, 16))

    # ── Separador ───────────────────────────
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=30)

    # ── Tabs ────────────────────────────────
    style = ttk.Style()
    style.theme_use("default")
    style.configure("TNotebook",
                    background=BG, borderwidth=0,
                    tabmargins=[0, 8, 0, 0])
    style.configure("TNotebook.Tab",
                    background=PANEL, foreground=SUBTEXT,
                    font=FONT_SMALL, padding=[14, 6],
                    borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", BG)],
              font=[("selected", ("Courier New", 9, "bold"))])

    notebook = ttk.Notebook(root, style="TNotebook")
    notebook.pack(fill="both", expand=True,
                  padx=22, pady=12)

    tab_mrua(notebook)
    tab_parabolico(notebook)
    tab_newton(notebook)
    tab_plano(notebook)

    # ── Footer ──────────────────────────────
    tk.Label(root,
             text="Ingresa los valores → Presiona ▶ Simular → Ver gráficas",
             bg=BG, fg=SUBTEXT, font=FONT_SMALL).pack(pady=(0, 14))

    root.mainloop()


if __name__ == "__main__":
    main()
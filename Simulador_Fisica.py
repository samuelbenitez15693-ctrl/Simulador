"""
Simulador de Física I - Cinemática y Dinámica
Universidad - Física I
Todo en una sola ventana | Resuelve incógnitas | Paleta blanco/verde
Requiere: pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ─────────────────────────────────────────────
#  PALETA: BLANCO Y VERDE
# ─────────────────────────────────────────────
BG       = "#f5f9f5"          # fondo principal blanco-verdoso
PANEL    = "#ffffff"          # paneles blancos
ACCENT   = "#1a7a3c"          # verde oscuro principal
ACCENT2  = "#27ae60"          # verde medio
ACCENT3  = "#52c77d"          # verde claro
TEXT     = "#1a2e1a"          # texto oscuro
SUBTEXT  = "#4a6b4a"          # subtexto verde grisáceo
SUCCESS  = "#0d5c2e"          # verde resultado
BORDER   = "#b8dfc8"          # borde suave verde
BG_ENTRY = "#eaf5ee"          # fondo inputs
SIDEBAR  = "#1a7a3c"          # barra lateral verde oscuro
HEADER   = "#eaf7ef"          # header suave

FONT_TITLE  = ("Courier New", 26, "bold")
FONT_SUB    = ("Courier New", 14, "bold")
FONT_LABEL  = ("Courier New", 12)
FONT_SMALL  = ("Courier New", 11)
FONT_RESULT = ("Courier New", 12, "bold")
FONT_INPUT  = ("Courier New", 12)
FONT_BTN    = ("Courier New", 13, "bold")
FONT_TAB    = ("Courier New", 12, "bold")
FONT_SOLVE  = ("Courier New", 11, "italic")


# ─────────────────────────────────────────────
#  FÍSICA: CINEMÁTICA
# ─────────────────────────────────────────────

def cinematica_mrua(v0, a, t_max, angulo=0):
    ang_rad = math.radians(angulo)
    v0x = v0 * math.cos(ang_rad)
    v0y = v0 * math.sin(ang_rad)
    g   = 9.8

    t = np.linspace(0, t_max, 600)

    if angulo == 0:
        x  = v0x * t + 0.5 * a * t**2
        vx = v0x + a * t
        ax = np.full_like(t, a)
        return {"t": t, "x": x, "v": vx, "a": ax, "tipo": "MRUA"}
    else:
        x  = v0x * t
        y  = v0y * t - 0.5 * g * t**2
        vx = np.full_like(t, v0x)
        vy = v0y - g * t
        v  = np.sqrt(vx**2 + vy**2)
        idx = np.where(y < 0)[0]
        if len(idx) > 0:
            cut = idx[0]
            x, y, t, v = x[:cut], y[:cut], t[:cut], v[:cut]
        return {"t": t, "x": x, "y": y, "v": v, "tipo": "Parabólico",
                "x_max": x[-1] if len(x) > 0 else 0,
                "y_max": float(np.max(y)) if len(y) > 0 else 0}


def segunda_ley(masa, fuerzas_x, fuerzas_y, rozamiento=0, t_max=10):
    Fx = sum(fuerzas_x) - rozamiento
    Fy = sum(fuerzas_y)
    ax = Fx / masa
    ay = Fy / masa
    a_total = math.sqrt(ax**2 + ay**2)

    t  = np.linspace(0, t_max, 600)
    vx = ax * t
    vy = ay * t
    v  = np.sqrt(vx**2 + vy**2)
    x  = 0.5 * ax * t**2
    y  = 0.5 * ay * t**2

    return {"t": t, "ax": ax, "ay": ay, "a": a_total,
            "vx": vx, "vy": vy, "v": v, "x": x, "y": y,
            "Fx": Fx, "Fy": Fy}


# ─────────────────────────────────────────────
#  SOLVER DE INCÓGNITAS MRUA
# ─────────────────────────────────────────────

def resolver_mrua(v0=None, vf=None, a=None, t=None, x=None):
    """
    Dadas 3 de 5 variables, calcula las 2 faltantes.
    Ecuaciones MRUA:
      vf = v0 + a*t
      x  = v0*t + 0.5*a*t^2
      vf^2 = v0^2 + 2*a*x
      x  = 0.5*(v0+vf)*t
    """
    knowns = sum(v is not None for v in [v0, vf, a, t, x])
    if knowns < 3:
        return None, "Necesitas al menos 3 valores conocidos."

    try:
        # Intentar despejar con las ecuaciones disponibles
        # Caso: v0, a, t → vf, x
        if v0 is not None and a is not None and t is not None and vf is None and x is None:
            vf = v0 + a * t
            x  = v0 * t + 0.5 * a * t**2

        # Caso: v0, vf, t → a, x
        elif v0 is not None and vf is not None and t is not None and a is None and x is None:
            a = (vf - v0) / t
            x = 0.5 * (v0 + vf) * t

        # Caso: v0, vf, a → t, x
        elif v0 is not None and vf is not None and a is not None and t is None and x is None:
            if a == 0:
                return None, "Si a=0, el tiempo es indeterminado con solo v0 y vf."
            t = (vf - v0) / a
            x = 0.5 * (v0 + vf) * t

        # Caso: v0, a, x → vf, t
        elif v0 is not None and a is not None and x is not None and vf is None and t is None:
            disc = v0**2 + 2 * a * x
            if disc < 0:
                return None, "No hay solución real (discriminante negativo)."
            vf = math.sqrt(disc)
            if a != 0:
                t = (vf - v0) / a
            else:
                t = x / v0 if v0 != 0 else None

        # Caso: vf, a, t → v0, x
        elif vf is not None and a is not None and t is not None and v0 is None and x is None:
            v0 = vf - a * t
            x  = v0 * t + 0.5 * a * t**2

        # Caso: v0, vf, x → a, t
        elif v0 is not None and vf is not None and x is not None and a is None and t is None:
            a = (vf**2 - v0**2) / (2 * x)
            t = 2 * x / (v0 + vf) if (v0 + vf) != 0 else None

        # Caso: a, t, x → v0, vf
        elif a is not None and t is not None and x is not None and v0 is None and vf is None:
            v0 = (x - 0.5 * a * t**2) / t if t != 0 else None
            if v0 is not None:
                vf = v0 + a * t

        # Caso: vf, t, x → v0, a
        elif vf is not None and t is not None and x is not None and v0 is None and a is None:
            v0 = 2 * x / t - vf if t != 0 else None
            if v0 is not None:
                a = (vf - v0) / t

        # Caso: v0, t, x → vf, a  (5to conocido puede ser cualquiera)
        elif v0 is not None and t is not None and x is not None and a is None and vf is None:
            a  = 2 * (x - v0 * t) / t**2 if t != 0 else None
            if a is not None:
                vf = v0 + a * t

        else:
            # Combinación con 4+ knowns → solo verificar/completar
            if v0 is not None and a is not None and t is not None:
                if vf is None: vf = v0 + a * t
                if x  is None: x  = v0 * t + 0.5 * a * t**2
            elif v0 is not None and vf is not None and t is not None:
                if a is None: a = (vf - v0) / t
                if x is None: x = 0.5 * (v0 + vf) * t

        return {"v0": v0, "vf": vf, "a": a, "t": t, "x": x}, None

    except Exception as e:
        return None, f"Error al resolver: {e}"


# ─────────────────────────────────────────────
#  SOLVER TIRO PARABÓLICO (incógnitas)
# ─────────────────────────────────────────────

def resolver_parabolico(v0=None, angulo=None, x_max=None, y_max=None, t_vuelo=None):
    """Dadas algunas variables, calcula las faltantes."""
    g = 9.8
    try:
        resultados = {}

        if v0 is not None and angulo is not None:
            rad = math.radians(angulo)
            v0x = v0 * math.cos(rad)
            v0y = v0 * math.sin(rad)
            t_v  = 2 * v0y / g
            x_m  = v0x * t_v
            y_m  = v0y**2 / (2 * g)
            resultados = {"v0": v0, "angulo": angulo,
                          "t_vuelo": t_v, "x_max": x_m, "y_max": y_m,
                          "v0x": v0x, "v0y": v0y}

        elif x_max is not None and angulo is not None:
            rad = math.radians(angulo)
            v0_calc = math.sqrt(x_max * g / math.sin(2 * rad))
            v0x = v0_calc * math.cos(rad)
            v0y = v0_calc * math.sin(rad)
            t_v = 2 * v0y / g
            y_m = v0y**2 / (2 * g)
            resultados = {"v0": v0_calc, "angulo": angulo,
                          "t_vuelo": t_v, "x_max": x_max, "y_max": y_m,
                          "v0x": v0x, "v0y": v0y}

        elif y_max is not None and angulo is not None:
            rad = math.radians(angulo)
            v0y = math.sqrt(2 * g * y_max)
            v0_calc = v0y / math.sin(rad)
            v0x = v0_calc * math.cos(rad)
            t_v = 2 * v0y / g
            x_m = v0x * t_v
            resultados = {"v0": v0_calc, "angulo": angulo,
                          "t_vuelo": t_v, "x_max": x_m, "y_max": y_max,
                          "v0x": v0x, "v0y": v0y}

        elif v0 is not None and x_max is not None:
            # sin(2θ) = x_max * g / v0²
            val = x_max * g / v0**2
            if abs(val) > 1:
                return None, "No hay solución: el alcance es mayor al máximo posible."
            ang = math.degrees(0.5 * math.asin(val))
            rad = math.radians(ang)
            v0x = v0 * math.cos(rad)
            v0y = v0 * math.sin(rad)
            t_v = 2 * v0y / g
            y_m = v0y**2 / (2 * g)
            resultados = {"v0": v0, "angulo": ang,
                          "t_vuelo": t_v, "x_max": x_max, "y_max": y_m,
                          "v0x": v0x, "v0y": v0y}
        else:
            return None, "Combinación de datos insuficiente o no soportada."

        return resultados, None
    except Exception as e:
        return None, f"Error: {e}"


# ─────────────────────────────────────────────
#  HELPER: ENTRADA CON LABEL
# ─────────────────────────────────────────────

def make_entry(parent, label, default="", width=12, hint=""):
    frame = tk.Frame(parent, bg=PANEL)
    frame.pack(fill="x", pady=4)

    lbl_frame = tk.Frame(frame, bg=PANEL)
    lbl_frame.pack(side="left", fill="x", expand=True)

    tk.Label(lbl_frame, text=label, bg=PANEL, fg=TEXT,
             font=FONT_LABEL, anchor="w").pack(side="left")
    if hint:
        tk.Label(lbl_frame, text=f"  {hint}", bg=PANEL, fg=ACCENT3,
                 font=FONT_SOLVE, anchor="w").pack(side="left")

    var = tk.StringVar(value=default)
    ent = tk.Entry(frame, textvariable=var, bg=BG_ENTRY,
                   fg=TEXT, font=FONT_INPUT, width=width,
                   relief="solid", bd=1, insertbackground=ACCENT,
                   highlightthickness=2,
                   highlightcolor=ACCENT,
                   highlightbackground=BORDER)
    ent.pack(side="right", padx=(8, 0))
    return var


def get_val(var):
    """Devuelve float o None si está vacío/inválido."""
    s = var.get().strip()
    if s == "" or s == "?":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL: VENTANA ÚNICA
# ─────────────────────────────────────────────

class SimuladorFisica:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Física I")
        self.root.configure(bg=BG)
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)

        self._build_ui()

    # ── UI PRINCIPAL ─────────────────────────
    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=ACCENT, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="⚛  FÍSICA I", bg=ACCENT, fg="white",
                 font=FONT_TITLE).pack(side="left", padx=28, pady=10)
        tk.Label(header, text="Simulador · Cinemática · Dinámica · Resolutor de Incógnitas",
                 bg=ACCENT, fg="#c8f0d8", font=FONT_SMALL).pack(side="left", pady=10)

        # ── Layout principal: sidebar + contenido ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # SIDEBAR de pestañas
        self.sidebar = tk.Frame(main, bg=SIDEBAR, width=190)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="MÓDULOS", bg=SIDEBAR, fg="#c8f0d8",
                 font=("Courier New", 11, "bold")).pack(pady=(20, 8))
        tk.Frame(self.sidebar, bg="#2d9e5c", height=1).pack(fill="x", padx=16)

        # Panel de contenido (derecha)
        self.content = tk.Frame(main, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)

        # Panel dividido: formulario | gráfica
        self.left_panel = tk.Frame(self.content, bg=PANEL,
                                   width=430, relief="flat", bd=0)
        self.left_panel.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self.left_panel.pack_propagate(False)

        self.right_panel = tk.Frame(self.content, bg=PANEL,
                                    relief="flat", bd=0)
        self.right_panel.pack(side="left", fill="both", expand=True,
                              padx=(6, 12), pady=12)

        # Matplotlib figure en el panel derecho
        plt.style.use("default")
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 4),
                                           facecolor=PANEL)
        self.fig.subplots_adjust(left=0.1, right=0.97,
                                 top=0.88, bottom=0.15, wspace=0.38)
        for ax in self.axes:
            ax.set_facecolor(BG_ENTRY)
            ax.tick_params(colors=SUBTEXT, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=8, pady=8)

        # Área de resultados (parte inferior del panel izquierdo)
        self.result_frame = tk.Frame(self.left_panel, bg=BG_ENTRY,
                                     relief="solid", bd=1)

        # Crear pestañas en sidebar
        self.tabs = {}
        self.tab_frames = {}
        opciones = [
            ("📐  MRUA", "mrua"),
            ("🎯  Tiro Parabólico", "parabolico"),
            ("⚡  2ª Ley Newton", "newton"),
        ]
        self.btn_tabs = {}
        for label, key in opciones:
            btn = tk.Button(self.sidebar, text=label, bg=SIDEBAR, fg="white",
                            font=("Courier New", 12), relief="flat", bd=0,
                            anchor="w", padx=18, pady=10, cursor="hand2",
                            activebackground=ACCENT2, activeforeground="white",
                            command=lambda k=key: self._switch_tab(k))
            btn.pack(fill="x")
            self.btn_tabs[key] = btn

        tk.Frame(self.sidebar, bg="#2d9e5c", height=1).pack(fill="x",
                                                             padx=16, pady=8)
        tk.Label(self.sidebar,
                 text="💡 Deja en blanco\nlas incógnitas\na resolver",
                 bg=SIDEBAR, fg="#a8dfc0", font=("Courier New", 10),
                 justify="center").pack(pady=6)

        # Construir contenido de cada pestaña
        self._build_mrua()
        self._build_parabolico()
        self._build_newton()

        # Activar primera pestaña
        self._switch_tab("mrua")

    # ── SWITCH DE PESTAÑA ────────────────────
    def _switch_tab(self, key):
        for k, frame in self.tab_frames.items():
            frame.pack_forget()
        for k, btn in self.btn_tabs.items():
            btn.configure(bg=SIDEBAR, fg="white")

        self.tab_frames[key].pack(fill="both", expand=True, padx=16, pady=12)
        self.btn_tabs[key].configure(bg=ACCENT2, fg="white")
        self._clear_graph()
        self._clear_result()

    def _clear_graph(self):
        for ax in self.axes:
            ax.cla()
            ax.set_facecolor(BG_ENTRY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
        self.canvas.draw()

    def _clear_result(self):
        self.result_frame.pack_forget()
        for w in self.result_frame.winfo_children():
            w.destroy()

    # ── MOSTRAR RESULTADO ────────────────────
    def _show_result(self, lines):
        self._clear_result()
        self.result_frame.pack(fill="x", padx=8, pady=(8, 8))

        tk.Label(self.result_frame, text="▼  RESULTADOS",
                 bg=BG_ENTRY, fg=ACCENT, font=FONT_SUB).pack(
                     anchor="w", padx=10, pady=(8, 4))

        txt = tk.Text(self.result_frame, bg=BG_ENTRY, fg=SUCCESS,
                      font=FONT_RESULT, relief="flat", bd=4,
                      wrap="word", height=len(lines) + 1,
                      state="normal")
        for line in lines:
            txt.insert("end", line + "\n")
        txt.configure(state="disabled")
        txt.pack(padx=8, pady=(0, 8), fill="x")

    # ── GRAFICAS ─────────────────────────────
    def _plot(self, datos, modo, titulo):
        for ax in self.axes:
            ax.cla()
            ax.set_facecolor(BG_ENTRY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.tick_params(colors=SUBTEXT, labelsize=9)

        self.fig.suptitle(titulo, color=ACCENT, fontsize=13,
                          fontfamily="Courier New", fontweight="bold")
        t = datos["t"]

        if modo == "mrua":
            self.axes[0].plot(t, datos["x"], color=ACCENT, lw=2.5)
            self.axes[0].fill_between(t, 0, datos["x"],
                                      alpha=0.12, color=ACCENT)
            self.axes[0].set_title("Posición vs Tiempo", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[0].set_xlabel("t (s)", color=SUBTEXT, fontsize=10)
            self.axes[0].set_ylabel("x (m)", color=SUBTEXT, fontsize=10)
            self.axes[0].grid(alpha=0.3, color=BORDER)

            self.axes[1].plot(t, datos["v"], color=ACCENT2, lw=2.5)
            self.axes[1].set_title("Velocidad vs Tiempo", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=10)
            self.axes[1].set_ylabel("v (m/s)", color=SUBTEXT, fontsize=10)
            self.axes[1].grid(alpha=0.3, color=BORDER)

        elif modo == "parabolico":
            self.axes[0].plot(datos["x"], datos["y"], color=ACCENT, lw=2.5)
            self.axes[0].fill_between(datos["x"], 0, datos["y"],
                                       alpha=0.12, color=ACCENT)
            self.axes[0].set_title("Trayectoria", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[0].set_xlabel("x (m)", color=SUBTEXT, fontsize=10)
            self.axes[0].set_ylabel("y (m)", color=SUBTEXT, fontsize=10)
            self.axes[0].grid(alpha=0.3, color=BORDER)

            self.axes[1].plot(t, datos["v"], color=ACCENT2, lw=2.5)
            self.axes[1].set_title("Velocidad vs Tiempo", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=10)
            self.axes[1].set_ylabel("|v| (m/s)", color=SUBTEXT, fontsize=10)
            self.axes[1].grid(alpha=0.3, color=BORDER)

        elif modo == "newton":
            self.axes[0].plot(t, datos["x"], color=ACCENT, lw=2.5)
            self.axes[0].fill_between(t, 0, datos["x"],
                                       alpha=0.12, color=ACCENT)
            self.axes[0].set_title("Posición vs Tiempo", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[0].set_xlabel("t (s)", color=SUBTEXT, fontsize=10)
            self.axes[0].set_ylabel("x (m)", color=SUBTEXT, fontsize=10)
            self.axes[0].grid(alpha=0.3, color=BORDER)

            self.axes[1].plot(t, datos["v"], color=ACCENT2, lw=2.5)
            self.axes[1].set_title("Velocidad vs Tiempo", color=ACCENT,
                                    fontsize=11, fontfamily="Courier New")
            self.axes[1].set_xlabel("t (s)", color=SUBTEXT, fontsize=10)
            self.axes[1].set_ylabel("v (m/s)", color=SUBTEXT, fontsize=10)
            self.axes[1].grid(alpha=0.3, color=BORDER)

        self.canvas.draw()

    # ─────────────────────────────────────────
    #  PESTAÑA: MRUA
    # ─────────────────────────────────────────
    def _build_mrua(self):
        frame = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["mrua"] = frame

        tk.Label(frame, text="Cinemática MRUA",
                 bg=PANEL, fg=ACCENT, font=FONT_SUB).pack(anchor="w",
                                                           pady=(0, 2))
        tk.Label(frame, text="x = v₀·t + ½·a·t²     vf = v0 + a·t",
                 bg=PANEL, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w",
                                                              pady=(0, 8))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

        tk.Label(frame,
                 text="",
                 bg=PANEL, fg=ACCENT2, font=("Courier New", 11, "italic")
                 ).pack(anchor="w", pady=(0, 6))

        self.mrua_v0 = make_entry(frame, "Vel. inicial  v₀  (m/s)", hint="")
        self.mrua_vf = make_entry(frame, "Vel. final    vf  (m/s)")
        self.mrua_a  = make_entry(frame, "Aceleración    a  (m/s²)")
        self.mrua_t  = make_entry(frame, "Tiempo         t  (s)")
        self.mrua_x  = make_entry(frame, "Desplazamiento x  (m)")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=10)

        tk.Button(frame, text="▶  Calcular y Simular",
                  bg=ACCENT, fg="white", font=FONT_BTN, relief="flat", bd=0,
                  padx=16, pady=10, cursor="hand2",
                  activebackground=ACCENT2, activeforeground="white",
                  command=self._simular_mrua).pack(fill="x")

        tk.Button(frame, text="↺  Limpiar",
                  bg=BORDER, fg=TEXT, font=FONT_LABEL, relief="flat", bd=0,
                  padx=10, pady=6, cursor="hand2",
                  command=lambda: self._limpiar_mrua()
                  ).pack(fill="x", pady=(6, 0))

    def _limpiar_mrua(self):
        for v in [self.mrua_v0, self.mrua_vf, self.mrua_a,
                  self.mrua_t, self.mrua_x]:
            v.set("")
        self._clear_graph()
        self._clear_result()

    def _simular_mrua(self):
        v0 = get_val(self.mrua_v0)
        vf = get_val(self.mrua_vf)
        a  = get_val(self.mrua_a)
        t  = get_val(self.mrua_t)
        x  = get_val(self.mrua_x)

        res, err = resolver_mrua(v0, vf, a, t, x)
        if err:
            messagebox.showerror("Error al resolver", err)
            return

        v0r = res["v0"]; vfr = res["vf"]
        ar  = res["a"];  tr  = res["t"]
        xr  = res["x"]

        # Validar que se resolvió todo
        vals = [v0r, vfr, ar, tr, xr]
        if any(v is None for v in vals):
            messagebox.showwarning("Incompleto",
                "No fue posible resolver todas las incógnitas con los datos dados.")
            return

        if tr <= 0:
            messagebox.showwarning("Advertencia",
                "El tiempo debe ser positivo.")
            return

        lines = [
            f"  v₀  = {v0r:.2f} m/s",
            f"  vf  = {vfr:.2f} m/s",
            f"  a   = {ar:.2f} m/s²",
            f"  t   = {tr:.2f} s",
            f"  x   = {xr:.2f} m",
            "",
            f"  Verif: vf = {v0r:.2f} + {ar:.2f}×{tr:.2f}",
            f"       = {v0r + ar*tr:.2f} m/s  ✔",
        ]
        self._show_result(lines)

        # Graficar con los valores resueltos
        datos = cinematica_mrua(v0r, ar, tr)
        self._plot(datos, "mrua", "MRUA – Cinemática")

    # ─────────────────────────────────────────
    #  PESTAÑA: TIRO PARABÓLICO
    # ─────────────────────────────────────────
    def _build_parabolico(self):
        frame = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["parabolico"] = frame

        tk.Label(frame, text="Tiro Parabólico",
                 bg=PANEL, fg=ACCENT, font=FONT_SUB).pack(anchor="w",
                                                           pady=(0, 2))
        tk.Label(frame, text="y = v₀·sin(θ)·t − ½·g·t²    g = 9.8 m/s²",
                 bg=PANEL, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w",
                                                               pady=(0, 8))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

        tk.Label(frame,
                 text="",
                 bg=PANEL, fg=ACCENT2,
                 font=("Courier New", 11, "italic")).pack(anchor="w",
                                                           pady=(0, 6))

        self.par_v0    = make_entry(frame, "Vel. inicial  v₀  (m/s)")
        self.par_ang   = make_entry(frame, "Ángulo        θ   (°)")
        self.par_xmax  = make_entry(frame, "Alcance       xₘₐₓ (m)")
        self.par_ymax  = make_entry(frame, "Alt. máxima   yₘₐₓ (m)")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=10)

        tk.Button(frame, text="▶  Calcular y Simular",
                  bg=ACCENT, fg="white", font=FONT_BTN, relief="flat", bd=0,
                  padx=16, pady=10, cursor="hand2",
                  activebackground=ACCENT2, activeforeground="white",
                  command=self._simular_parabolico).pack(fill="x")

        tk.Button(frame, text="↺  Limpiar",
                  bg=BORDER, fg=TEXT, font=FONT_LABEL, relief="flat", bd=0,
                  padx=10, pady=6, cursor="hand2",
                  command=lambda: [v.set("") for v in
                                   [self.par_v0, self.par_ang,
                                    self.par_xmax, self.par_ymax]]
                  ).pack(fill="x", pady=(6, 0))

    def _simular_parabolico(self):
        v0   = get_val(self.par_v0)
        ang  = get_val(self.par_ang)
        xmax = get_val(self.par_xmax)
        ymax = get_val(self.par_ymax)

        res, err = resolver_parabolico(v0, ang, xmax, ymax)
        if err:
            messagebox.showerror("Error al resolver", err)
            return

        v0r   = res["v0"]
        angr  = res["angulo"]
        xmr   = res["x_max"]
        ymr   = res["y_max"]
        tvr   = res["t_vuelo"]
        v0xr  = res["v0x"]
        v0yr  = res["v0y"]

        if not (0 < angr < 90):
            messagebox.showwarning("Advertencia",
                "El ángulo debe estar entre 0° y 90°.")
            return

        lines = [
            f"  v₀        = {v0r:.2f} m/s",
            f"  θ         = {angr:.2f}°",
            f"  v₀ₓ       = {v0xr:.2f} m/s",
            f"  v₀ᵧ       = {v0yr:.2f} m/s",
            f"  Alcance   = {xmr:.2f} m",
            f"  Alt. máx  = {ymr:.2f} m",
            f"  T. vuelo  = {tvr:.2f} s",
        ]
        self._show_result(lines)

        datos = cinematica_mrua(v0r, 0, tvr * 1.05, angulo=angr)
        self._plot(datos, "parabolico", "Tiro Parabólico")

    # ─────────────────────────────────────────
    #  PESTAÑA: 2ª LEY DE NEWTON
    # ─────────────────────────────────────────
    def _build_newton(self):
        frame = tk.Frame(self.left_panel, bg=PANEL)
        self.tab_frames["newton"] = frame

        tk.Label(frame, text="Segunda Ley de Newton",
                 bg=PANEL, fg=ACCENT, font=FONT_SUB).pack(anchor="w",
                                                           pady=(0, 2))
        tk.Label(frame, text="ΣF = m · a",
                 bg=PANEL, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w",
                                                              pady=(0, 8))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

        tk.Label(frame,
                 text="💡 Deja en blanco la incógnita (m, F o a):",
                 bg=PANEL, fg=ACCENT2,
                 font=("Courier New", 11, "italic")).pack(anchor="w",
                                                          pady=(0, 6))

        self.newt_masa = make_entry(frame, "Masa          m   (kg)", "")
        self.newt_fx   = make_entry(frame, "Fuerza aplic. Fₓ  (N)", "")
        self.newt_ff   = make_entry(frame, "Rozamiento    Ff  (N)", "0")
        self.newt_a    = make_entry(frame, "Aceleración   a   (m/s²)", "")
        self.newt_t    = make_entry(frame, "Tiempo simul. t   (s)", "8")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=10)

        tk.Button(frame, text="▶  Calcular y Simular",
                  bg=ACCENT, fg="white", font=FONT_BTN, relief="flat", bd=0,
                  padx=16, pady=10, cursor="hand2",
                  activebackground=ACCENT2, activeforeground="white",
                  command=self._simular_newton).pack(fill="x")

        tk.Button(frame, text="↺  Limpiar",
                  bg=BORDER, fg=TEXT, font=FONT_LABEL, relief="flat", bd=0,
                  padx=10, pady=6, cursor="hand2",
                  command=lambda: [v.set("") for v in
                                   [self.newt_masa, self.newt_fx,
                                    self.newt_ff, self.newt_a]]
                  ).pack(fill="x", pady=(6, 0))

    def _simular_newton(self):
        m   = get_val(self.newt_masa)
        fx  = get_val(self.newt_fx)
        ff  = get_val(self.newt_ff) or 0.0
        a   = get_val(self.newt_a)
        tm  = get_val(self.newt_t) or 8.0

        # Resolver incógnita
        if m is None and fx is not None and a is not None:
            Fnet = fx - ff
            if a == 0:
                messagebox.showerror("Error", "Si a=0 la masa es indeterminada.")
                return
            m = Fnet / a
        elif fx is None and m is not None and a is not None:
            fx = m * a + ff
        elif a is None and m is not None and fx is not None:
            Fnet = fx - ff
            a = Fnet / m
        elif m is not None and fx is not None:
            Fnet = fx - ff
            a = Fnet / m
        else:
            messagebox.showerror("Error",
                "Ingresa al menos 2 de: Masa, Fuerza, Aceleración.")
            return

        if m <= 0:
            messagebox.showwarning("Error", "La masa debe ser positiva.")
            return

        datos = segunda_ley(m, [fx], [0], rozamiento=ff, t_max=tm)
        datos["v"] = datos["vx"]

        lines = [
            f"  Masa          = {m:.2f} kg",
            f"  Fuerza aplic. = {fx:.2f} N",
            f"  Rozamiento    = {ff:.2f} N",
            f"  Fuerza neta   = {fx - ff:.2f} N",
            f"  Aceleración   = {datos['ax']:.2f} m/s²",
            "",
            f"  Verif: a = ΣF/m = {fx-ff:.2f}/{m:.2f}",
            f"           = {datos['ax']:.2f} m/s²  ✔",
        ]
        self._show_result(lines)
        self._plot({"t": datos["t"], "x": datos["x"], "v": datos["vx"]},
                   "newton", "2ª Ley de Newton")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = SimuladorFisica(root)
    root.mainloop()


if __name__ == "__main__":
    main()
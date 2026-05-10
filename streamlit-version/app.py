import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
# ═══════════════════════════════════════════════════════════════════
#CONFIGURACION DE LA PAGINA
#═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Simulador de Movimiento I",
    page_icon=":rocket:",
    layout="centered",
)
# ═══════════════════════════════════════════════════════════════════
#ESTILOS
#═══════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
           .main {
                background-color: #f0f0f0;
                color: white;
           }

              .stButton>button {
                background-color: #4CAF50;
                color: black;
                font-weight: bold;
                border-radius: 10px;
                height: 3em;
                width: 100%;
            }

            .stNumberInput label{
                color: white !important;
            }

            h1,h2, h3 {
                color: #00FF00;
            }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#FUNCIONES FISICAS
#═══════════════════════════════════════════════════════════════════    

def cinematica_mrua(v0, a, t_max):
    t=np.linspace(0, t_max, 500)
    x=v0*t + 0.5*a*t**2
    v = v0 + a*t
    return {
        't': t,
        'x': x,
        'v': v,
        'a': np.full_like(t, a)
    }
def resolver_mrua(v0=None, vf=None, a=None, t =None, x=None):
    if sum(v is not None for v in [v0,vf,a,t,x]) < 3:
        return None, "Error: Se necesitan al menos 3 variables para resolver el MRUA.",[]
    pasos = []

    try:
        #-----------------------
        #CASO 1
        #-----------------------
        if v0 is not None and a is not None and t is not None:
            vf = v0 + a*t
            x = v0*t + 0.5*a*t**2
            pasos.append("Usamos la ecuacion: ")
            pasos.append("vf = v0 + a*t")

            pasos.append(f"vf = {v0} + ({a})({t})"
            )

            pasos.append("")

            pasos.append("Usamos la ecuacion: ")
            pasos.append("x = v0*t + 0.5*a*t^2")

            pasos.append(f"x = {v0}*{t} + 0.5({a})({t}^2)"
            )

            pasos.append(
                f"x = {x:.2f} m"
            )
        #------------------------------
        #CASO 2
        #------------------------------
        elif v0 is not None and vf is not None and t is not None:
            a = (vf - v0) / t
            x = 0.5*(v0 + vf) * t
            pasos.append("Usamos: ")
            pasos.append("a = (vf - v0) / t")
            pasos.append(f"a = ({vf} - {v0}) / {t}")
            pasos.append( f"a = {a:.2f} m/s^2" )
            pasos.append("")
            pasos.append("Usamos: ")
            pasos.append("x = 0.5*(v0 + vf) * t")
            pasos.append(f"x = 0.5*({v0} + {vf}) * ({t})")
            pasos.append(f"x = {x:.2f} m")
            
        #------------------------------
        #CASO 3
        #------------------------------
        elif v0 is not None and vf is not None  and a is not None:
            t = (vf - v0) / a
            x = 0.5*(v0 + vf) * t
            pasos.append("Usamos: ")
            pasos.append("t = (vf - v0) / a")
            pasos.append(f"t = ({vf} - {v0}) / {a}")
            pasos.append( f"t = {t:.2f} s" )
            pasos.append("")
            pasos.append("Usamos: ")
            pasos.append("x = 0.5*(v0 + vf) * t")
            pasos.append(f"x = 0.5*({v0} + {vf}) * ({t:.2f})")
            pasos.append(f"x = {x:.2f} m")
            
        #------------------------------
        #CASO 4
        #------------------------------
        elif v0 is not None and a is not None and x is not None:
            descriminante = v0**2 + 2*a*x
            if descriminante < 0:
                return None, "Error: No hay solución real para el tiempo con los valores dados.", pasos
            vf = math.sqrt(descriminante)
            t = (vf - v0) / a
            pasos.append("Usamos: ")
            pasos.append("vf = v0^2 + 2*a*x")
            pasos.append(
                f"vf^2 = { v0**2} + 2({a})({x})"
            )
            pasos.append(f"vf = {vf:.2f} m/s")
            pasos.append("")
            pasos.append("Usamos: ")
            pasos.append("t = (vf - v0) / a")
            pasos.append(f"t = ({vf:.2f} - {v0}) / {a}")
            pasos.append(f"t = {t:.2f} s")

        else :
            return None, "Combinaciuon todavia no implementada", []
        return {
            'v0': v0,
            'vf': vf,
            'a': a,
            't': t,
            'x': x
        }, None, pasos
    except Exception as e:
        return None, str(e), []
    

# ═══════════════════════════════════════════════════════════════════
#FUNCIONES TIRO PARABOLICO
#═══════════════════════════════════════════════════════════════════
def resolver_parabolico(v0, angulo):
    g = 9.8
    rad = math.radians(angulo)
    v0x = v0 * math.cos(rad)
    v0y = v0 * math.sin(rad)
    tiempo = (2 * v0y) / g
    alcance = v0x * tiempo
    altura = (v0y**2) / (2 * g)
    return {
        'v0x': v0x,
        'v0y': v0y,
        'tiempo': tiempo,
        'alcance': alcance,
        'altura': altura
    }
def trayectoria_parabolica(v0, angulo):
    g = 9.8
    rad = math.radians(angulo)
    v0x = v0 * math.cos(rad)
    v0y = v0 * math.sin(rad)
    tiempo_total = (2 * v0y) / g
    t = np.linspace(0, tiempo_total, 500)
    x = v0x * t
    y = v0y * t - 0.5 * g * t**2
    vy = v0y - g*t
    velocidad = np.sqrt(v0x**2 + vy**2)
    return {
        't': t,
        'x': x,
        'y': y,
        "v" : velocidad
    }

# ═══════════════════════════════════════════════════════════════════
#SIDEBAR
#═══════════════════════════════════════════════════════════════════
st.sidebar.title("Simulador")

modulo = st.sidebar.radio(
    "Seleciona un modulo:",
    [
        "MRUA",
        "Tiro Parabolico"
        ]
)

# ═══════════════════════════════════════════════════════════════════
#MODULO MRUA
# ═══════════════════════════════════════════════════════════════════

if modulo == "MRUA":
    st.title("Movimiento Rectilineo Uniformewnente Variado")

    st.mardown("""
               ### Formulas
               - vf = v0 + a*t
               - x = v0*t + 0.5*a*t^2
               - vf^2 = v0^2 + 2*a*x
               -x = 0.5*(v0 + vf) * t
               """)
    col1, col2 = st.columnms(2)
    with col1:
        v0= st.number_input(
            "Velocidad inicial v0 (m/s)",
            value = None,
            placeholder = "Ej: 10"
        )

        vf= st.number_input(
            "Velocidad final vf (m/s)",
            value = None,
            placeholder = "Ej: 30"
        )
        a= st.number_input(
            "Aceleracion a (m/s^2)",
            value = None,
            placeholder = "Ej: 2"
        )
    with col2:
        t= st.number_input(
            "Tiempo t (s)",
            value = None,
            placeholder = "Ej: 5"
        )
        x= st.number_input(
            "Distancia x (m)",
            value = None,
            placeholder = "Ej: 100"
        )
        calcular = st.button("Calcular MRUA")
    if calcular:
        datos, error, pasos = resolver_mrua(v0, vf, a, t, x)
        if error:
            st.error(error)
        else:
            st.success("Calculo completado")
            c1,c2,c3,c5, = st.columns(5)

            with c1:
                st.metric("Velocidad Inicial", f"{datos['v0']:.2f} m/s")
            with c2:
                st.metric("Velocidad Final", f"{datos['vf']:.2f} m/s")
            with c3:
                st.metric("Aceleracion", f"{datos['a']:.2f} m/s^2")
            with c5:
                st.metric("x", f"{datos['x']:.2f} m")
            
            st.markdown("### Pasos del calculo")
            for paso in pasos:
                st.write(paso) 

            sim = cinematica_mrua(
                datos["v0"],
                datos["a"],
                datos["t"]
            )
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            fig.patch.set_facecolor("black")

            axs[0, 0].plot(sim['t'], sim['x'], color='cyan')
            axs[0, 0].set_title("Posicion vs Tiempo", color='white')
            axs[0,0].grid(True)

            axs[0, 1].plot(sim['t'], sim['v'], color='magenta')
            axs[0, 1].set_title("Velocidad vs Tiempo", color='white')
            axs[0,1].grid(True)

            axs[1, 0].plot(sim['t'], sim['a'], color='yellow')
            axs[1,  0].set_title("Aceleracion vs Tiempo", color='white')
            axs[1,0].grid(True)

            axs[1, 1].plot(sim['x'], sim['v'], color='orange')
            axs[1, 1].set_title("Velocidad vs Posicion", color='white')
            axs[1,1].grid(True)

            for ax in axs.flat:
                ax.set_facecolor("black")
                ax.tick_params(colors='white')
                ax.title.set_color('white')

                for spine in ax.spines.values():
                    spine.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════
#MODULO TIRO PARABOLICO
# ═══════════════════════════════════════════════════════════════════

elif modulo == "Tiro Parabolico":
    st.title("Tiro Parabolico")

    st.markdown("""
    ### Formulas
    - x = v0x * t
    - y = v0y * t - 0.5 * g * t^2
    - Alcance = v0^2 * sin(2*angulo) / g
    - Altura = v0^2 * sin^2(angulo) / (2*g)
    """)
    col1, col2 = st.columns(2)
    with col1:
        v0 = st.number_input(
            "Velocidad inicial v0 (m/s)",
            min_value=0.0,
            value=20.0
        )
    with col2:
        angulo = st.slider(
            "Angulo de lanzamiento (grados)",
            1,
            89,
            45
        )
    calcular_par = st.button("Simular Tiro")

    if calcular_par:
        resultados = resolver_parabolico(v0, angulo)
        st.success("Simulacion completada")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric(
                "Tiempo de vuelo", f"{resultados['tiempo']:.2f} s"
            )
        with c2:
            st.metric(
                "Alcance", f"{resultados['alcance']:.2f} m"
            )
        with c3:
            st.metric(
                "Altura maxima", f"{resultados['altura']:.2f} m"
            )
        
        st.markdown("## Solucion paso a paso")
        st.write("## Componentes de velocidad")
        st.write(f"v0x = v0 * cos(angulo)")
        st.write(
            f"v0x = {resultados['v0x']:.2f} m/s"
        )
        st.write(
            f"v0y = v0 * sin(angulo)"
        )
        st.write(f"v0y = {resultados['v0y']:.2f} m/s")
        st.write("")
        st.write("## Tiempo de vuelo")
        st.write(
            "tiempo = (2 * v0y) / g"
        )
        st.write(
            f"tiempo = (2 * {resultados['v0y']:.2f}) / 9.8"
        )
        st.write(
            f"tiempo = {resultados['tiempo']:.2f} s"
        )
        st.write("")
        st.write("## Alcance")
        st.write(
            "Alcance = v0x * tiempo"
        )
        st.write(
            f"Alcance = {resultados['v0x']:.2f} * {resultados['tiempo']:.2f}"
        )
        st.write(
            f"Alcance = {resultados['alcance']:.2f} m"
        )
        st.write("")
        st.write("## Altura maxima")
        st.write(
            "Altura = (v0y^2) / (2 * g)"
        )
        st.write(
            f"Altura = ({resultados['v0y']:.2f}^2) / (2 * 9.8)"
        )
        st.write(
            f"Altura = {resultados['altura']:.2f} m"
        )

        datos = trayectoria_parabolica(v0, angulo)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("black")
        axs[0,0].plot(
            datos["x"],
            datos["y"]
        )
        axs[0,0].set_title("Trayectoria del proyectil", color='white')
        axs[0,0].grid(True)

        axs[0,1].plot(
            datos["t"],
            datos["v"]
        )
        axs[0,1].set_title("Velocidad vs Tiempo", color='white')
        axs[0,1].grid(True)
        axs[1,0].plot(
            datos["t"],
            datos["x"]
        )
        axs[1,0].set_title("Posición Horizontal", color='white')
        axs[1,0].grid(True)
        axs[1,1].plot(
            datos["t"],
            datos["y"]
        )
        axs[1,1].set_title("Altura")
        axs[1,1].grid(True)

        for ax in axs.flat:
            ax.set_facecolor("black")
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

            for spine in ax.spines.values():
                spine.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════
#FOOTER
# ═══════════════════════════════════════════════════════════════════
st.marckdown(""" Proyecto de Fisica I """)

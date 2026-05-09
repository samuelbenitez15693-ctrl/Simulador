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
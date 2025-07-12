"""
Frontend Streamlit de *Bias Sense*.

- Permite al usuario pegar un texto y detectar posibles sesgos.
"""
import html
import requests
import streamlit as st
from typing import List, Dict

# ---------- GLOBAL CSS FOR BETTER HIGHLIGHT ----------
st.markdown(
    """
    <style>
    /* nicer mark formatting so highlights don’t look cut */
    mark {
        padding: 0.15em 0.25em;
        border-radius: 4px;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CONFIG & SIDEBAR ----------
st.sidebar.header("⚙️ Configuración")

# URL por defecto (puede venir de .streamlit/secrets.toml)
try:
    DEFAULT_API_URL = st.secrets["API_URL"]
except Exception:
    DEFAULT_API_URL = ""

api_url = st.sidebar.text_input("API endpoint", value=DEFAULT_API_URL)

# Lista de modos mostrados en el selector
MODES = ("Demo • sin spans", "Demo • con spans", "Backend real")

# Si el usuario escribe una URL válida, forzamos “Backend real”
forced_mode = "Backend real" if api_url else MODES[0]

mode = st.sidebar.selectbox("Modo de ejecución", MODES, index=MODES.index(forced_mode))

DEMO_MODE: bool = mode.startswith("Demo")
DEMO_SPANS: bool = mode.endswith("con spans")

if DEMO_MODE:
    st.sidebar.info(f"Ejecutando en **{mode}**")
else:
    st.sidebar.success("Modo backend real activado")

LABEL_COLORS = {
    "gender": "#F4A6B7",
    "race": "#8CC2F2",
    "politics": "#A4D4AE",
    "hate": "#FFD86E",
    "offensive": "#FFB980",
    "other": "#D3D3D3",
}

# ---------- HELPER ----------
def call_api(text: str) -> Dict:
    """
    Devuelve un diccionario con:
    - labels: probabilidades por sesgo
    - spans: posiciones a resaltar
    - neutral_text: versión reescrita sin sesgo

    El contenido depende del modo de ejecución seleccionado.
    """
    # --- DEMO: sin spans ---
    if DEMO_MODE and not DEMO_SPANS:
        return {
            "labels": {
                "gender": 0.15,
                "race": 0.05,
                "politics": 0.60,
                "hate": 0.02,
                "offensive": 0.08,
            },
            "spans": [],
            "neutral_text": text,
        }

    # --- DEMO: con spans de ejemplo ---
    if DEMO_MODE and DEMO_SPANS:
        dummy_spans = [
            {"start": 4, "end": 36, "label": "politics"},
            {"start": 55, "end": 74, "label": "gender"},
        ]
        return {
            "labels": {
                "gender": 0.45,
                "race": 0.03,
                "politics": 0.70,
                "hate": 0.01,
                "offensive": 0.05,
            },
            "spans": dummy_spans,
            "neutral_text": "Versión neutral generada con LLM…",
        }

    # --- BACKEND real ---
    resp = requests.post(api_url, json={"text": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def highlight_text(text: str, spans: List[Dict]) -> str:
    """
    Inserta marcas <mark> sobre las posiciones indicadas en *spans*.

    📌 Futuro: cuando el modelo use LIME/SHAP para explicar la predicción,
    simplemente rellenaremos la lista *spans* con los índices producidos
    por el explicador.
    """
    spans_sorted = sorted(spans, key=lambda s: s["start"])
    html_parts, cursor = [], 0
    for span in spans_sorted:
        start, end, label = span["start"], span["end"], span["label"]
        html_parts.append(html.escape(text[cursor:start]))
        html_parts.append(
            f"<mark style='background:{LABEL_COLORS.get(label, '#FFFFCC')}'>{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    html_parts.append(html.escape(text[cursor:]))
    return "".join(html_parts)

# ---------- UI ----------
st.title("Bias Sense Demo")
st.markdown("Ingresa un texto en inglés para detectar posibles sesgos:")

user_text = st.text_area("Texto de entrada", height=200)
if st.button("Analizar sesgos") and user_text.strip():
    with st.spinner("Analizando…"):
        try:
            result = call_api(user_text)
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")
        else:
            # ----- Scores -----
            st.subheader("Probabilidad por sesgo")
            sorted_scores = sorted(result["labels"].items(), key=lambda x: x[1], reverse=True)
            # Si todas las probabilidades son muy bajas, avisamos y saltamos el resto
            MAX_PROB = max(result["labels"].values()) if result["labels"] else 0
            if MAX_PROB < 0.10:
                st.info("No se detectaron sesgos relevantes en el texto ingresado.")
            else:
                for lbl, prob in sorted_scores:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.markdown(f"**{lbl}**")
                    with c2:
                        st.progress(prob)
                        st.write(f"{prob:.1%}")

            # ----- Text highlight -----
            st.subheader("Texto con frases sesgadas resaltadas")
            if MAX_PROB < 0.10:
                # Sin sesgos significativos ➜ no resaltamos nada
                st.info("—")
            else:
                if result.get("spans"):
                    # ✔️ La API envió spans → los usamos
                    html_text = highlight_text(user_text, result["spans"])
                    st.markdown(html_text, unsafe_allow_html=True)
                else:
                    # ❗️ La API NO envió spans → mostramos el texto tal cual y avisamos
                    st.warning(
                        "La API no devolvió fragmentos específicos para resaltar "
                    )
                    st.write(user_text)

            # ----- Neutral text -----
            st.subheader("Versión neutral sugerida")
            neutral = result["neutral_text"]
            st.success(neutral)
            if st.button("Copiar texto neutral"):
                st.session_state["_copy"] = neutral

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Bias Sense • v0.2 • © 2025")

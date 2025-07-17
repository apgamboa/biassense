"""
Frontend Streamlit de *Bias Sense*.

- Permite al usuario pegar un texto y detectar posibles sesgos.
"""
import html
import requests
import streamlit as st
from typing import List, Dict

# ---------- CONFIG ----------
api_url_naive_bayes: str = "http://localhost:8080/predict"
api_url_transformer: str = "http://localhost:8081/predict_transformer"

# ---------- MODEL VERSION SELECTOR ----------
MODEL_CHOICES = {
    "Naive Bayes (clásico)": "nb",          # old scikit‑learn model
    "Transformer (multilabel)": "transformer"  # new HF‑embeddings model
}
model_key = st.sidebar.radio(
    "Modelo a utilizar",
    list(MODEL_CHOICES.keys()),
    index=0  # pre‑select Naive Bayes
)
model_version = MODEL_CHOICES[model_key]

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

LABEL_COLORS = {
    "gender": "#F4A6B7",
    "gender_bias": "#F4A6B7",
    "race": "#8CC2F2",
    "social_bias": "#8CC2F2",
    "politics": "#A4D4AE",
    "political_bias": "#A4D4AE",
    "hate": "#FFD86E",
    "hate_speech": "#FFD86E",
    "religion_bias": "#C7B8FF",
    "offensive": "#FFB980",
    "other": "#D3D3D3",
}

# ---------- HELPER ----------
def call_api(text: str) -> Dict:
    """
    Llama al backend FastAPI correspondiente (Naive Bayes o Transformer)
    y devuelve su JSON.
    """
    if model_version == "nb":
        url = api_url_naive_bayes
    else:  # "transformer"
        url = api_url_transformer

    resp = requests.post(url, json={"text": text}, timeout=50)
    resp.raise_for_status()
    return resp.json()

def highlight_text(text: str, spans: List[Dict]) -> str:
    """
    Inserta marcas <mark> sobre las posiciones indicadas en *spans*.
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
            if not result["labels"]:
                st.info("No se recibieron probabilidades desde la API.")
            else:
                MAX_PROB = max(result["labels"].values())
                for lbl, prob in sorted_scores:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.markdown(f"**{lbl}**")
                    with c2:
                        st.progress(prob)
                        st.write(f"{prob:.1%}")

            # ----- Text highlight -----
            st.subheader("Texto con frases sesgadas resaltadas")
            if not result["labels"]:
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
st.caption("Bias Sense • v1.0 • © 2025")

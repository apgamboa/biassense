from pathlib import Path
import uvicorn, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re

from bias_sense.data_layer.data import generate_preprocess_new_text
from bias_sense.utils.utilities import load_pickle

import google.generativeai as genai
from dotenv import load_dotenv
import os

# ─── Gemini setup ────────────────────────────────────────────────────────────
load_dotenv()  # loads API_KEY_GOOGLE from .env
genai.configure(api_key=os.getenv("API_KEY_GOOGLE"))

def text_generator(biased_text: str) -> str:
    """
    Rewrites *biased_text* into a neutral, inclusive version using Gemini 2.5‑Flash.
    Returns only the rewritten text (no extra comments).
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        "Provide ONLY the rewritten text, without any introductory comments.\n"
        "Rewrite the following text to make it completely neutral, inclusive, and free from discriminatory biases. "
        "Ensure to:\n"
        "- Remove sexist language (e.g., use 'people' instead of 'men').\n"
        "- Avoid stereotypes about gender, ethnicity, age, etc.\n"
        "- Use disability‑inclusive language (e.g., 'people with disabilities').\n"
        "- Remain culturally & religiously neutral.\n"
        "- Prefer non‑binary / inclusive terms.\n"
        "- Keep an objective tone, avoiding subjective adjectives or value judgements.\n\n"
        "Text to rewrite:\n"
    )
    response = model.generate_content(prompt + biased_text)
    # Gemini returns a Content object → take first candidate text
    return response.candidates[0].content.parts[0].text.strip()

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "cloud_training"

# ─── carga única al arrancar ────────────────────────────────────────────────────
cv = load_pickle(ARTIFACTS, "count_vectorizer_std.pickle")
mod_bias      = load_pickle(ARTIFACTS, "model_bias.pickle")
mod_sentiment = load_pickle(ARTIFACTS, "model_sentiment.pickle")
mod_label     = load_pickle(ARTIFACTS, "model_label.pickle")

cat_bias      = pd.read_csv(ARTIFACTS / "catalog_values_encoded_bias.csv")
cat_sentiment = pd.read_csv(ARTIFACTS / "catalog_values_encoded_sentiment.csv")
cat_label     = pd.read_csv(ARTIFACTS / "catalog_values_encoded_label.csv")

# Build class‑name list dynamically from catalog
class_names = cat_bias["target"].tolist()
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(
    class_names=class_names,
    bow=True,
    split_expression=r"\W+"
)

if any(x is None for x in (cv, mod_bias, mod_sentiment, mod_label)):
    raise RuntimeError("❌  Faltan artifacts: entrena primero con bias_sense/main.py")

def bias_predict_proba(texts: list[str] | np.ndarray) -> np.ndarray:

    X_vec = cv.transform(texts)
    return mod_bias.predict_proba(X_vec)

# ─── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Bias-Sense API (v0)")

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextIn):
    try:
        X = generate_preprocess_new_text(payload.text, cv)

        pred_bias      = mod_bias.predict(X)[0]
        pred_sentiment = mod_sentiment.predict(X)[0]
        pred_label     = mod_label.predict(X)[0]

        label_bias      = cat_bias.loc[cat_bias.encoded_target == pred_bias, "target"].item()
        label_sentiment = cat_sentiment.loc[cat_sentiment.encoded_target == pred_sentiment, "target"].item()
        label_label     = cat_label.loc[cat_label.encoded_target == pred_label, "target"].item()

        prob_bias      = float(mod_bias.predict_proba(X)[0][pred_bias])
        prob_sentiment = float(mod_sentiment.predict_proba(X)[0][pred_sentiment])
        prob_label     = float(mod_label.predict_proba(X)[0][pred_label])

        spans = lime_to_spans(
            payload.text,
            bias_predict_proba,
            pred_bias,
            label_bias,
            top_k=5
        )

        return {
            "labels": {
                label_bias:      prob_bias,
                label_sentiment: prob_sentiment,
                label_label:     prob_label
            },
            "spans": spans,
            "neutral_text": text_generator(payload.text)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Ejecutar con `python -m api.main` opcional ────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)

def lime_to_spans(text: str,
                  predict_fn,
                  class_id: int,
                  label_name: str,
                  top_k: int = 5) -> list[dict]:
    """
    Ejecuta LIME y devuelve los rangos (start,end,label) de hasta top_k tokens
    con peso positivo para la clase `class_id`.
    """
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        labels=[class_id],
        num_features=top_k * 2,      # pedimos de sobra
        num_samples=500              # ajusta si es lento
    )
    pos_tokens = [tok for tok, w in explanation.as_list(label=class_id) if w > 0][:top_k]

    spans = []
    for tok in pos_tokens:
        # primera ocurrencia de cada token (case-insensitive)
        m = re.search(re.escape(tok), text, re.I)
        if m:
            spans.append({"start": m.start(),
                          "end":   m.end(),
                          "label": label_name})
    return spans

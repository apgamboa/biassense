from pathlib import Path
import re, uvicorn, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer

# --- carga modelo ---
from bias_sense.models.multilabel_transformer import BiasClassifier
from bias_sense.models.gen_ai import text_generator

classifier     = BiasClassifier()
predict_proba  = lambda xs: classifier.predict_proba(xs)
class_names    = list(classifier._label2idx)          # ['other', 'social_bias', ...]
explainer      = LimeTextExplainer(class_names=class_names, bow=True, split_expression=r"\W+")

# probabilidad mínima para mostrar spans (ajustado ↓ para textos menos extremos)
THRESH = 0.10
TOP_K  = 5        # tokens por clase

app = FastAPI(title="Bias-Sense API – Transformer")

class TextIn(BaseModel):
    text: str

def lime_spans(text: str, class_id: int, label: str) -> list[dict]:
    exp = explainer.explain_instance(text, predict_proba, labels=[class_id],
                                     num_features=TOP_K*2, num_samples=500)
    pos = [tok for tok,w in exp.as_list(label=class_id) if w>0][:TOP_K]
    spans=[]
    for tok in pos:
        m = re.search(re.escape(tok), text, re.I)
        if m:
            spans.append(dict(start=m.start(), end=m.end(), label=label))
    return spans

@app.post("/predict_transformer")
def predict_tf(payload: TextIn):
    try:
        probs = classifier.predict(payload.text).to_dict()   # 6 floats
        spans=[]
        for lbl, p in probs.items():
            if p >= THRESH:
                cid = classifier._label2idx[lbl]
                spans.extend(lime_spans(payload.text, cid, lbl))
        return dict(
            labels = probs,
            spans  = spans,
            neutral_text = text_generator(payload.text)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.api_transformer:app", host="0.0.0.0", port=8081, reload=True)

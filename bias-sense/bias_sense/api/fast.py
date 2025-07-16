from fastapi import FastAPI, Query
from bias_sense.main_transformer import get_bias
from typing import Any

app = FastAPI(
    title="BiasSense Transformer API",
    description="API para detección de sesgo usando el modelo multilabel transformer",
    version="1.0.0",
)

@app.get("/")
def root() -> dict[str, str]:
    """
    Endpoint raíz de salud.
    """
    return {"greeting": "BiasSense Transformer API está viva!"}

@app.get("/predict")
def predict(text: str = Query(..., description="Texto a analizar")) -> Any:
    """
    Endpoint de predicción.
    Llama a `get_bias(text)` y devuelve el resultado como JSON.
    """
    result = get_bias(text)
    # Si `result` tiene método to_dict(), lo usamos; si ya es dict, lo devolvemos
    try:
        return result.to_dict()
    except AttributeError:
        return result

"""Multilabel transformer‑based bias detector.

Incluye:
- Dataclass de salida `BiasDetectionResult`.
- Cliente de embeddings `InfinityEmbClient` con soporte batch (feature_extraction).
- `BiasClassifier` que usa TF + emb client y expone `predict` y `predict_proba`.

Pensado para usarse desde `api/api_transformer.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib
import requests
import tensorflow as tf

# ---------------------------------------------------------------------------
# Salida estructurada --------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BiasDetectionResult:
    other: float
    social_bias: float
    hate_speech: float
    gender_bias: float
    political_bias: float
    religion_bias: float

    def to_dict(self) -> dict:  # handy for JSON
        return asdict(self)

# ---------------------------------------------------------------------------
# Cliente de embeddings (Infinity / HF provider) ----------------------------
# ---------------------------------------------------------------------------

class InfinityEmbClient:
    """Wrapea el endpoint Infinity‑Embeddings con reuse de sesión y batch.

    Parameters
    ----------
    token : str
        Bearer token válido para el provider.
    model : str, default "Qwen/Qwen3-Embedding-8B"
        Nombre del modelo remoto.
    """

    _URL = (
        "https://vicenciojulio2025--qwen-infinity-serve-infinity.modal.run/embeddings"
    )

    def __init__(self, token: str, model: str = "Qwen/Qwen3-Embedding-8B") -> None:
        self._session = requests.Session()
        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",  # usa el token recibido
            "Content-Type": "application/json",
        }
        self._model = model

    # --- batch (preferido) --------------------------------------------------
    def feature_extraction(self, texts: List[str]) -> List[List[float]]:
        """Devuelve embedding por cada texto en una sola llamada HTTP."""
        payload: Dict[str, Any] = {
            "model": self._model,
            "encoding_format": "float",
            "dimensions": 0,  # usar dims por defecto (4096 en este modelo)
            "input": texts,
            "modality": "text",
        }
        resp = self._session.post(
            self._URL, headers=self._headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

    # --- un texto (retro‑compat) -------------------------------------------
    def embed(self, text: str) -> List[float]:
        return self.feature_extraction([text])[0]

# ---------------------------------------------------------------------------
# Clasificador ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class BiasClassifier:
    """Carga pesos Keras + binarizer y predice prob. de sesgo."""

    _EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

    def __init__(
        self,
        model_path: str | None = None,
        mlb_path: str | None = None,
        infinity_token: str | None = None,
    ) -> None:
        # localización de artifacts
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = model_path or project_root / "artifacts/deep_learning/modelo.h5"
        mlb_path = mlb_path or project_root / "artifacts/deep_learning/multilabel_binarizer.joblib"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        if not Path(mlb_path).exists():
            raise FileNotFoundError(f"No se encontró el binarizer: {mlb_path}")

        # carga pesos y binarizer
        self._model = tf.keras.models.load_model(model_path, compile=False)
        self._mlb = joblib.load(mlb_path)

        # cliente embeddings
        infinity_token = infinity_token or os.getenv("INFINITY_TOKEN")
        if not infinity_token:
            raise RuntimeError("Define INFINITY_TOKEN con tu Bearer‑token")
        self._emb_client = InfinityEmbClient(token=infinity_token, model=self._EMBED_MODEL)

        # map label -> idx para convertir probs
        self._label2idx = {lbl: i for i, lbl in enumerate(self._mlb.classes_)}

    # ------------------------- API pública ----------------------------------

    def predict(self, text: str) -> BiasDetectionResult:
        vec = self._emb_client.embed(text)  # list[float]
        probs = self._model.predict(np.asarray([vec], dtype="float32"), verbose=0)[0]
        return self._probs_to_result(probs)

    def predict_proba(self, texts: list[str] | str) -> np.ndarray:
        """Devuelve matriz (n_samples, n_labels) de probabilidades."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._emb_client.feature_extraction(texts)  # 🔹 1 sola llamada HTTP
        embs = np.asarray(vecs, dtype="float32")
        return self._model.predict(embs, verbose=0)

    # ------------------------- utils internos -------------------------------

    def _probs_to_result(self, probs: np.ndarray) -> BiasDetectionResult:
        bias_types = [
            "other",
            "social_bias",
            "hate_speech",
            "gender_bias",
            "political_bias",
            "religion_bias",
        ]
        ordered = [float(probs[self._label2idx[lbl]]) for lbl in bias_types]
        return BiasDetectionResult(*ordered)

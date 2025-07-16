from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf
from huggingface_hub import InferenceClient


# Objeto de salida para la API
@dataclass(slots=True)
class BiasDetectionResult:
    other: float
    social_bias: float
    hate_speech: float
    gender_bias: float
    political_bias: float
    religion_bias: float

    def to_dict(self) -> dict:
        """Para convertirlo a diccionario en vez de objeto"""
        return asdict(self)


# Clase del modelo
class BiasClassifier:
    _EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

    def __init__(
            self,
            model_path: str = None,
            mlb_path: str = None,
            hf_token: str | None = None,
    ) -> None:
        # Obtener el directorio del archivo actual (multilabel_transformer.py)
        current_file = Path(__file__).resolve()

        # multilabel_transformer.py está en bias_sense/models/
        # Necesitamos subir 2 niveles para llegar al directorio raíz del proyecto
        project_root = current_file.parent.parent.parent  # Esto nos lleva a bias-sense/

        # Construir las rutas
        if model_path is None:
            model_path = project_root / "artifacts" / "deep_learning" / "modelo.h5"
        if mlb_path is None:
            mlb_path = project_root / "artifacts" / "deep_learning" / "multilabel_binarizer.joblib"

        # Verificar que los archivos existen
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
        if not Path(mlb_path).exists():
            raise FileNotFoundError(f"No se encontró el archivo MLBinarizer: {mlb_path}")

        # 2.1 Pesos del modelo y binarizer
        self._model = tf.keras.models.load_model(model_path, compile=False)
        self._mlb = joblib.load(mlb_path)

        # 2.2 Cliente para pasar el texto input a vector de embeddings
        # Leer el token correcto y pasarlo como `api_key`
        hf_api_key = hf_token or os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            raise RuntimeError("Necesito HUGGINGFACE_API_KEY en el entorno para instanciar InferenceClient")
        # Ahora sí le pasamos el api_key que espera el cliente
        self._emb_client = InferenceClient(provider="nebius", api_key=hf_api_key)

        # 2.3 Mapeo de etiqueta a índice para construir el resultado
        self._label2idx = {lbl: i for i, lbl in enumerate(self._mlb.classes_)}

    # Método público para obtener el sesgo en un texto, este se usará en el main
    def predict(self, text: str) -> BiasDetectionResult:
        emb = self._text_to_embedding(text)
        probs = self._model.predict(emb, verbose=0)[0]  # (6,)
        return self._probs_to_result(probs)

    # Métodos de uso interno
    def _text_to_embedding(self, text: str) -> np.ndarray:
        vec = self._emb_client.feature_extraction(text, model=self._EMBED_MODEL)
        return np.asarray(vec, dtype="float32")  # .reshape(1, -1)

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

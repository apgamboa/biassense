from dotenv import load_dotenv
load_dotenv()   # carga .env y pone las vars en os.environ

import os
import json
import sys
from bias_sense.models.multilabel_transformer import BiasClassifier

_classifier: BiasClassifier | None = None


# singleton para no sobre instanciarlo
def _get_classifier() -> BiasClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BiasClassifier()
    return _classifier


def get_bias(text: str):
    """
    Devuelve BiasDetectionResult con los seis porcentajes (0-1).
    """
    return _get_classifier().predict(text)

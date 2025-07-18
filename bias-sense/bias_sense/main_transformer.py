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


if __name__ == "__main__":
    # Si se quiere usar en consola, pero lo dejo pendiente
    """
    if len(sys.argv) < 2:
        print("Uso: python -m bias_sense.main \"Texto a analizar\"")
        sys.exit(1)

    texto = " ".join(sys.argv[1:])
    resultado = get_bias(texto)
    print(json.dumps(resultado.to_dict(), ensure_ascii=False, indent=2))
    """

    texto = "¿Conocen el juego de Hitler secreto?"
    resultado = get_bias(texto)
    print(json.dumps(resultado.to_dict(), ensure_ascii=False, indent=2))
    print(resultado)

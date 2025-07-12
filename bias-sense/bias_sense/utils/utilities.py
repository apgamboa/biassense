import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage, bigquery
import pandas as pd
from bias_sense.params import GCP_PROJECT, BUCKET_NAME, ARTIFACTS_FOLDER

# Carga variables de entorno desde .env
load_dotenv()


def load_pickle(path: Path, filename: str):
    """
    Carga un objeto pickle desde <path>/<filename>.
    Devuelve el objeto o None si no existe o hay error.
    """
    full_path = path / filename
    if full_path.is_file():
        try:
            with full_path.open('rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_pickle(path: Path, filename: str, obj):
    """
    Guarda un objeto como pickle en <path>/<filename>.
    Devuelve el objeto guardado.
    """
    full_path = path / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with full_path.open('wb') as f:
        pickle.dump(obj, f)
    return obj


def upload_to_gcs(local_path):
    """
    Sube `local_path` (Path o str) a:
      gs://<BUCKET_NAME>/<ARTIFACTS_FOLDER>/<archivo>
    """
    local = Path(local_path)
    fname = local.name

    client = storage.Client(project=os.getenv('GCP_PROJECT') or GCP_PROJECT)
    bucket = client.bucket(os.getenv('BUCKET_NAME') or BUCKET_NAME)
    remote = f"{ARTIFACTS_FOLDER}/{fname}"

    bucket.blob(remote).upload_from_filename(str(local))
    print(f"✅ Subido {fname} → gs://{bucket.name}/{remote}")


def upload_all_pickles():
    """
    Escanea la carpeta raíz/artifacts y sube
    TODOS los archivos *.pickle (ignora .csv) llamando a upload_to_gcs().
    """
    project_root  = Path(__file__).parents[2]
    artifacts_dir = project_root / "artifacts"

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {artifacts_dir}")

    for pk in artifacts_dir.glob("*.pickle"):
        upload_to_gcs(pk)


# -- Chunking utilities ---------------------------------------------------

def query_bq_in_chunks(query: str, project: str, chunk_size: int):
    """
    Ejecuta la consulta en BigQuery y devuelve DataFrames de tamaño <= chunk_size.
    """
    client = bigquery.Client(project=project or os.getenv('GCP_PROJECT') or GCP_PROJECT)
    job = client.query(query)
    iterator = job.result(page_size=chunk_size)
    for page in iterator.pages:
        yield page.to_dataframe()


def read_csv_in_chunks(path: Path, file_name: str, chunksize: int):
    """
    Lee un CSV local en fragmentos de 'chunksize' filas.
    """
    full_path = path / file_name
    for df_chunk in pd.read_csv(full_path, chunksize=chunksize):
        yield df_chunk


def load_vectorizer(path: Path, filename: str = "count_vectorizer_std.pickle"):
    """Carga el CountVectorizer entrenado."""
    return load_pickle(path, filename)


def load_encoder(path: Path, filename: str = "encoder_label.pickle"):
    """Carga el LabelEncoder entrenado."""
    return load_pickle(path, filename)


def load_model(path: Path, filename: str = "model_bias.pickle"):
    """Carga el modelo de bias entrenado."""
    return load_pickle(path, filename)


def save_model(path: Path, model, filename: str = "model_bias.pickle"):
    """Guarda el modelo entrenado y devuelve la ruta."""
    save_pickle(path, filename, model)
    return path / filename

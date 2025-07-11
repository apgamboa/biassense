import os
import pickle
from pathlib import Path
from google.cloud import storage
from bias_sense.params import GCP_PROJECT, BUCKET_NAME, ARTIFACTS_FOLDER
from dotenv import load_dotenv
load_dotenv()


def load_pickle(path, file_name):
    full_path = path / file_name
    if os.path.isfile(full_path):
        with open(full_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception: # so many things could go wrong, can't be more specific.
                pass
    return None

def save_pickle(path, file_name, transformer_model):
    full_path = path / file_name
    with open(full_path, "wb") as f:
        pickle.dump(transformer_model, f)
    return transformer_model


def upload_to_gcs(path):
    """
    Sube path (Path o str) bajo ./artifacts/ a:
      gs://<BUCKET_NAME>/<ARTIFACTS_FOLDER>/<archivo>
    """
    local = Path(path)
    fname = local.name

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    blob   = bucket.blob(f"{ARTIFACTS_FOLDER}/{fname}")

    blob.upload_from_filename(str(local))
    # Con Uniform Bucket-Level Access, los objetos heredan permiso de bucket IAM
    print(f"✅ Subido {fname} → gs://{BUCKET_NAME}/{blob.name}")
    public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}"
    print(public_url)
    return public_url



def upload_all_pickles():
    """
     Escanea la carpeta <repo_root>/artifacts y sube
     TODOS los archivos *.pickle (ignora .csv) llamando a upload_to_gcs().
    """
    # Encuentra la raíz de “bias-sense/” (el directorio que contiene artifacts/)
    project_root  = Path(__file__).parents[2]
    artifacts_dir = project_root / "artifacts"

    if not artifacts_dir.exists():
         raise FileNotFoundError(f"No existe la carpeta: {artifacts_dir}")

     # Recorre sólo los pickles
    for pk in artifacts_dir.glob("*.pickle"):
         upload_to_gcs(pk)

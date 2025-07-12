import os
import numpy as np
from dotenv import load_dotenv, find_dotenv

from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Encuentra el .env en la raíz del repo (sube carpetas hasta dar con él)
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError("No se encontró un archivo .env en el árbol de carpetas")
# Cárgalo (override=True para asegurarnos de sobreescribir cualquier var previa)
load_dotenv(dotenv_path, override=True)

##################  VARIABLES  ##################
# Control de tamaños y chunking
# Lo leemos siempre como string porque luego lo validamos contra la lista
DATA_SIZE  = os.getenv("DATA_SIZE", "10000")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))  # 200 o más

# Dónde guardar el modelo
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")  # "local", "gcs", "mlflow"

# Google Cloud Projects
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_REGION  = os.getenv("GCP_REGION", "")

# BigQuery
BQ_DATASET = os.getenv("BQ_DATASET", "")
BQ_REGION  = os.getenv("BQ_REGION", "")

# Cloud Storage
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
ARTIFACTS_FOLDER = os.getenv("ARTIFACTS_FOLDER", "artifacts")  # Carpeta para subir PICKLEs y modelos

# Carpeta dentro del bucket donde están los CSV
RAW_FOLDER = os.getenv("RAW_FOLDER", "biassense-raw")

# Compute Engine (VM)
INSTANCE = os.getenv("INSTANCE", "")

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "")


#Google IA
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

##################  CONSTANTS  ####################
LOCAL_DATA_PATH     = os.path.join(os.path.expanduser("~"), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser("~"), ".lewagon", "mlops", "training_outputs")

COLUMN_NAMES_RAW = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

DTYPES_PROCESSED = np.float32

##################  VALIDATIONS  ####################
# Sólo tamaños que ya están en GCS
AVAILABLE_DATA_SIZES = ["10000", "20000", "50000", "100000", "500000", "full-data"]
if DATA_SIZE not in AVAILABLE_DATA_SIZES:
    raise ValueError(f"Invalid DATA_SIZE: '{DATA_SIZE}' not in {AVAILABLE_DATA_SIZES}")

# Sólo estos destinos de modelo
VALID_MODEL_TARGETS = ["local", "gcp", "mlflow"]
if MODEL_TARGET not in VALID_MODEL_TARGETS:
    raise ValueError(f"Invalid MODEL_TARGET: '{MODEL_TARGET}' not in {VALID_MODEL_TARGETS}")

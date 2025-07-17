import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Encuentra y carga el .env
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError("No se encontró un archivo .env en el árbol de carpetas")
load_dotenv(dotenv_path, override=True)

##################  VARIABLES  ##################
# Control de tamaños y chunking
DATA_SIZE  = os.getenv("DATA_SIZE", "10000")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))

# Destino del modelo: "local", "gcp" o "mlflow"
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")

# Google Cloud Projects
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_REGION  = os.getenv("GCP_REGION", "")

# BigQuery
BQ_DATASET = os.getenv("BQ_DATASET", "")
BQ_REGION  = os.getenv("BQ_REGION", "")

# Cloud Storage
BUCKET_NAME      = os.getenv("BUCKET_NAME", "")
ARTIFACTS_FOLDER = os.getenv("ARTIFACTS_FOLDER", "artifacts")

# Carpeta dentro del bucket donde están los CSV
RAW_FOLDER = os.getenv("RAW_FOLDER", "biassense-raw")

# Compute Engine (VM)
INSTANCE = os.getenv("INSTANCE", "")

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "")

##################  GOOGLE GENERATIVE AI ##################
# Clave y modelo por defecto
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")

if not GOOGLE_API_KEY:
    raise ValueError("Falta GOOGLE_API_KEY en tu .env")

# Configura la SDK de Gemini / PaLM 2
genai.configure(api_key=GOOGLE_API_KEY)

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

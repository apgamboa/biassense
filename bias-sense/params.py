import os
import numpy as np

##################  VARIABLES  ##################
# Control de tamaños y chunking
DATA_SIZE              = os.environ.get("DATA_SIZE")           #20000,50000,100000,200000,500000
CHUNK_SIZE             = int(os.environ.get("CHUNK_SIZE"))     # 200 o mas

# Dónde guardar el modelo
MODEL_TARGET           = os.environ.get("MODEL_TARGET")        # "local", "gcs"

# Rango para evaluación
#VALUATION_START_DATE  = os.environ.get("EVALUATION_START_DATE")  # "YYYY-MM-DD"

# Google Cloud Projects
GCP_PROJECT            = os.environ.get("GCP_PROJECT")          # biassense
GCP_REGION             = os.environ.get("GCP_REGION")           # "us-central1"

# BigQuery
BQ_DATASET             = os.environ.get("BQ_DATASET")           # "biassense_dataset"
BQ_REGION              = os.environ.get("BQ_REGION")            # "EU"

# Cloud Storage
BUCKET_NAME            = os.environ.get("BUCKET_NAME")          # "csdev_biassense_rr"

# Compute Engine (VM)
INSTANCE               = os.environ.get("INSTANCE")             # vm-biassense

MLFLOW_TRACKING_URI    = os.environ.get("MLFLOW_TRACKING_URI")  # ej. https://mlflow.lewagon.ai"
MLFLOW_EXPERIMENT      = os.environ.get("MLFLOW_EXPERIMENT")    # taxifare_experiment_jfro57
MLFLOW_MODEL_NAME      = os.environ.get("MLFLOW_MODEL_NAME")    # taxifare_jfro57


##################  CONSTANTS  (Aun Le falta esta parte) ####################
LOCAL_DATA_PATH       = os.path.join(
    os.path.expanduser("~"), ".lewagon", "mlops", "data"
)
LOCAL_REGISTRY_PATH   = os.path.join(
    os.path.expanduser("~"), ".lewagon", "mlops", "training_outputs"
)

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

################## VALIDATIONS   Aca se va a modificar todavía ####################
env_valid_options = dict(
    DATA_SIZE=["1k", "200k", "all"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ.get(env)
    if env_value not in valid_options:
        raise ValueError(
            f"Invalid value for {env} in `.env`: '{env_value}' not in {valid_options}"
        )

for env, valid_opts in env_valid_options.items():
    validate_env_value(env, valid_opts)

import os
import pandas as pd
import numpy as np
import os.path
import re
from colorama import Fore, Style
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
# Importando la nube de BQ
from google.cloud import bigquery

# Carga las variables de .env (incluye ENV)
load_dotenv()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean raw data by
    - filtering only the useful tags
    - fixing text format and length
    '''
    df = df.dropna(subset=['aspect'])
    df = df.reset_index(drop=True)

    # Limpieza de texto
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    df['cleaned_text'] = df['cleaned_text'].astype(str)\
        .apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).strip())
    df['cleaned_text'] = df['cleaned_text']\
        .str.replace(r'[^\w\s]|_','', regex=True)
    df['cleaned_text'] = df['cleaned_text']\
        .str.replace(r'\s+', ' ', regex=True).str.strip()
    df['cleaned_text'] = df['cleaned_text']\
        .str.replace(r'\d+', '', regex=True)
    df['cleaned_text'] = df['cleaned_text']\
        .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    df['cleaned_text'] = df['cleaned_text']\
        .str.replace(r'\b(ha|lo)(ha|l)+\b', '', regex=True, flags=re.IGNORECASE)\
        .str.strip()

    lemmatizer = WordNetLemmatizer()
    df['tokenized_text'] = [
        word_tokenize(text, language='english')
        for text in df['cleaned_text']
    ]
    df['lemmatized_text'] = df['tokenized_text']\
        .apply(lambda tokens: [lemmatizer.lemmatize(token, pos='v') for token in tokens])
    df['lemmatized_text'] = df['lemmatized_text']\
        .apply(lambda tokens: ' '.join(tokens))

    print("✅ data cleaned")
    return df

def preprocess_features(X: pd.DataFrame, count_vectorizer) -> pd.DataFrame:
    text_vectorized = count_vectorizer.transform(X['lemmatized_text'])
    return pd.DataFrame(
        text_vectorized.toarray(),
        columns = count_vectorizer.get_feature_names_out(),
        index = X['lemmatized_text']
    )

def create_vectorizer(X: pd.DataFrame, target: str):
    count_vectorizer = CountVectorizer(max_df=0.9, min_df=8)
    count_vectorizer.fit(X[target])
    return count_vectorizer

def encoding_feature(X: pd.DataFrame, feature: str):
    label_encoder = LabelEncoder()
    label_encoder.fit(X[feature])
    encoded_target = label_encoder.transform(X[feature])
    catalog_values_encoded = pd.DataFrame({
        "target": X[feature],
        "encoded_target": encoded_target
    }).drop_duplicates()
    return encoded_target, catalog_values_encoded

def get_data() -> pd.DataFrame:
    """
    Lee datos desde archivo CSV local.
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: get_data (local)" + Style.RESET_ALL)
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "../../data/reduced_dataset.csv")
    data = pd.read_csv(path)
    print("✅ get_data() done \n")
    return data

def get_data_bq() -> pd.DataFrame:
    """
    Lee los datos de BigQuery cuando ENV='gcp'.
    """
    print("☁️ get_data_bq()")
    print("→ GCP_PROJECT =", os.environ.get("GCP_PROJECT"))
    project = os.environ["GCP_PROJECT"]
    dataset = os.environ["BQ_DATASET"]
    table   = os.environ.get("BQ_TABLE", "reduced_dataset")
    client = bigquery.Client(project=project, location='EU')
    query = f"""
        SELECT *
        FROM `{project}.{dataset}.{table}`
    """
    data = client.query(query).result().to_dataframe()
    print("✅ get_data_bq done — filas:", len(data))
    return data

def fetch_data() -> pd.DataFrame:
    """
    Wrapper que elige entre local o GCP según ENV.
    """
    env = os.getenv("ENV", "local").lower()
    if env == "gcp":
        return get_data_bq()
    else:
        return get_data()

def generate_preprocess_new_text(text: str, count_vectorizer):
    """
    - Get new X preprocess to predict
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: generate_preprocess_new_text" + Style.RESET_ALL)
    data = {'cleaned_text': [text], 'aspect': ["NA"]}
    data = pd.DataFrame(data)
    data_clean = clean_data(data)
    X_processed = preprocess_features(data_clean, count_vectorizer)
    print("✅ generate_preprocess_new_text() done \n")
    return X_processed

def get_sample_data(data: pd.DataFrame):
    """
    - Get sample data to train y evaluar modelo
    """
    sample = 10000
    print(Fore.MAGENTA + "\n ⭐️ Use case: get_sample_data" + Style.RESET_ALL)
    data = data.sample(sample)
    print("✅ get_sample_data() done \n")
    return data

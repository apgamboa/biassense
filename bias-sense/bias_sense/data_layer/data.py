import pandas as pd
import numpy as np
import os.path
import re
from colorama import Fore, Style
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

##MIgrar esto a la clase utitlities cuando funcione
import os
import pickle


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

##MIgrar esto a la clase utitlities cuando funcione


#Importando
from google.cloud import bigquery


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean raw data by
    -filtering only the useful tags
    -fixing text format and lenght
    '''
    # Compress raw_data by setting types to DTYPES_RAW
    #TBC

    #Remove NA
    df = df.dropna(subset=['aspect'])
   #df = df.dropna(subset=['text'])
    df = df.reset_index(drop=True)

    #Cleans text and creates a new column
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).strip()) #Delete URLs, mentions and hashtags
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^\w\s]|_','', regex=True) #Delete . and _
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip() #Normalizes spaces
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\d+', '', regex=True) #delete numbers
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True) #delete symbols
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\b(ha|lo)(ha|l)+\b', '', regex=True, flags=re.IGNORECASE) #delete laugh expresions, common in df
    df['cleaned_text'] = df['cleaned_text'].str.strip() #delete spaces at beggining and end of text

    lemmatizer = WordNetLemmatizer()

    #tokenize and lemmatize, preparing for vectorization
    df['tokenized_text'] = [word_tokenize(text, language='english') for text in df['cleaned_text']]
    df['lemmatized_text'] = df['tokenized_text'].apply(lambda tokens: [lemmatizer.lemmatize(token, pos='v') for token in tokens])
    df['lemmatized_text'] = df['lemmatized_text'].apply(lambda tokens: ' '.join([''.join(t) if isinstance(t, list) else t for t in tokens]))

    print("✅ data cleaned")

    return df

def preprocess_features(X: pd.DataFrame, count_vectorizer) -> pd.DataFrame:

    text_vectorized = count_vectorizer.transform(X['lemmatized_text'])
    # Show the representations in a nice DataFrame
    text_vectorized = pd.DataFrame(
        text_vectorized.toarray(),
        columns = count_vectorizer.get_feature_names_out(),
        index = X['lemmatized_text']
    )

    return text_vectorized


def create_vectorizer(X: pd.DataFrame, target:str):

     # Vectorize the sentences
    count_vectorizer = CountVectorizer(ngram_range = (1,2), max_df = 0.9, min_df=8)
    #count_vectorizer = CountVectorizer(max_df = 0.9, min_df=8)

    #text_vectorized = count_vectorizer.fit_transform(X['lemmatized_text'])
    count_vectorizer.fit(X[target])

    return count_vectorizer

def encoding_feature(X: pd.DataFrame, feature: str):

    # Instantiate the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit it to the target
    label_encoder.fit(X[feature])

    # Transform the targets
    encoded_target = label_encoder.transform(X[feature])

    catalog_values_encoded =  pd.DataFrame({"target": X[feature], "encoded_target": encoded_target}).drop_duplicates()


    return encoded_target, catalog_values_encoded

def get_data():
    """
    -Get all data from csv, it could be google bigquery
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: get_data" + Style.RESET_ALL)

    #Query data meanwhile import from local file
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "../../data/reduced_dataset.csv")
    data = pd.read_csv(path)

    print("✅ get_data() done \n")

    return data

def generate_preprocess_new_text(text:str, count_vectorizer):

    """
    -Get new X preprocess to predict
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: generate_preprocess_new_text" + Style.RESET_ALL)

    # Process data
    data = {
    'cleaned_text': [text],
    'aspect':["NA"]
    }
    data = pd.DataFrame(data)
    data_clean = clean_data(data)
    X_processed = preprocess_features(data_clean, count_vectorizer)

    print("✅ generate_preprocess_new_text() done \n")

    return X_processed

def get_sample_data(data:pd.DataFrame):
    """
    -Get sample data to train and evaluate model
    """
    sample = 10000
    print(Fore.MAGENTA + "\n ⭐️ Use case: get_sample_data" + Style.RESET_ALL)
    #print(data)
    # Train an use a portion of dataset
    data = data.sample(sample)
    # Delete nas
    #data["lemmatized_text"] = data["lemmatized_text"].fillna(' ')

    print("✅ get_sample_data() done \n")

    return data

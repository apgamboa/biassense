import pandas as pd
import numpy as np
import re
from colorama import Fore, Style
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


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
    df = df.dropna(subset=['text'])
    df = df.reset_index(drop=True)

    #Cleans text and creates a new column
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    df['cleaned_text'] = df['text'].astype(str).apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).strip()) #Delete URLs, mentions and hashtags
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^\w\s]|_','', regex=True) #Delete . and _
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip() #Normalizes spaces
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\d+', '', regex=True) #delete numbers
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True) #delete symbols
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\b(ha|lo)(ha|l)+\b', '', regex=True, flags=re.IGNORECASE) #delete laugh expresions, common in df
    df['cleaned_text'] = df['cleaned_text'].strip() #delete spaces at beggining and end of text

    lemmatizer = WordNetLemmatizer()

    #tokenize and lemmatize, preparing for vectorization
    df['tokenized_text'] = [word_tokenize(text, language='english') for text in df['cleaned_text']]
    df['lemmatized_text'] = df['tokenized_text'].apply(lambda tokens: [lemmatizer.lemmatize(token, pos='v') for token in tokens])
    df['lemmatized_text'] = df['lemmatized_text'].apply(lambda tokens: ' '.join([''.join(t) if isinstance(t, list) else t for t in tokens]))

    print("✅ data cleaned")

    return df

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn process that transforms a cleaned dataset
        into a preprocessed one.
        """
        final_preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(max_features=5000, min_df=8, max_df=0.95), 'lemmatized_text'),
        ('cat', OneHotEncoder(), ['aspect', 'label', 'sentiment','identity_mention']),
        ('num', StandardScaler(), 'toxic')
        ])

        return final_preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed

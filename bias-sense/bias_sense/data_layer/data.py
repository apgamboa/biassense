import pandas as pd
import re
from nltk.stem import WordNetLemmatizer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean raw data by
    -filtering only the useful tags
    -fixing text format and lenght
    '''
    # Compress raw_data by setting types to DTYPES_RAW
    #TBC

    #Remove NA
    df_filtered = df_filtered.dropna(subset=['aspect'])
    df_filtered = df_filtered.dropna(subset=['text'])
    df_filtered = df_filtered.reset_index(drop=True)

    #Cleans text and creates a new column
    df_filtered['cleaned_text'] = df_filtered['text'].astype(str).apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x).strip())
    df_filtered['cleaned_text'] = df_filtered['cleaned_text'].str.lower()
    df_filtered['cleaned_text'] = df_filtered['cleaned_text'].str.replace(r'[^\w\s]|_','', regex=True)
    df_filtered['cleaned_text'] = df_filtered['cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df_filtered['cleaned_text'] = df_filtered['cleaned_text'].str.replace(r"'", '', regex=True)

    #Tokenizing
    lemmatizer = WordNetLemmatizer()



    #Prepares final dataset
    clean_df = df_filtered.drop(columns=['text', 'dimension'])

    return clean_df

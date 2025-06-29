import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from colorama import Fore, Style

from bias_sense.data_layer import clean_data, preprocess_features


def preprocess():
    """
    -Cleans and preprocess data
    Future: store processed data on BQ
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    #Query data meanwhile import from local file

    df = pd.read_csv('reduced_dataset.csv')

    # Process data
    data_clean = clean_data(df)

    X = data_clean.drop(columns=['bias_type','tokenized_text', 'cleaned_text'])
    y = data_clean['bias_type']

    X_processed = preprocess_features(X)

    label_encoded = LabelEncoder()
    y_encoded = label_encoded.fit(y)

    print("✅ preprocess() done \n")


def train():
    pass


def evaluate():
    pass

def pred():
    pass

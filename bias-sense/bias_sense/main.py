import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from bias_sense.models.model import naive_bayes, naive_bayes_to_evaluate
from sklearn.naive_bayes import MultinomialNB


from colorama import Fore, Style

from bias_sense.data_layer.data import clean_data, preprocess_features, create_vectorizer, encoding_feature
from bias_sense.data_layer.data import get_data, generate_preprocess_new_text, get_sample_data


def preprocess(data:pd.DataFrame, target:str):
    """
    -Cleans and preprocess data
    Future: store processed data on BQ
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    data_clean = clean_data(data)

    # Create X, y
    X = data_clean.drop(columns=['tokenized_text', 'cleaned_text'])

    #Create transformer
    count_vectorizer = create_vectorizer(X, target)

    # transform or process data
    X_processed = preprocess_features(X, count_vectorizer)

    # Encode y
    #y_encoded, catalog_values_encoded = encoding_features(y,target)


    print("✅ preprocess() done \n")

    return X_processed, count_vectorizer


def train(X:pd.DataFrame,y:np.array):

    """
    -Train dataset
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: train" + Style.RESET_ALL)

    model = naive_bayes(X, y)
    model.fit(X, y)

    print("✅ train() done \n")

    return model


def evaluate(X:pd.DataFrame,y:np.array):
    """
    -Evaluate model and get results
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = naive_bayes_to_evaluate()

    # Cross-validation
    metrics = cross_validate(model, X, y, cv = 5)

    print("✅ evaluate() done \n")


    return metrics


def pred(X:pd.DataFrame, model: MultinomialNB):
    """
    -Predictions
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    y_pred = model.predict(X)

    print("✅ pred() done \n")
    ## Falta codificar el regreso
    return y_pred


if __name__ == '__main__':
    try:
        #Create data set
        text = "This is a text to test in the model a tendency of religion or god"
        data = get_data()
        data_sample = get_sample_data(data)

        ##Train the model and predict bias
        X, count_vectorizer_std = preprocess(data_sample,'lemmatized_text') # preprocess data and get transformer, and catalog
        y_encoded_bias, catalog_values_encoded_bias = encoding_feature(data_sample,'bias_type')
        results_bias = evaluate(X, y_encoded_bias)
        print(results_bias)
        model_bias = train(X,y_encoded_bias)
        X_new_bias = generate_preprocess_new_text(text, count_vectorizer_std)
        y_result_bias = pred(X_new_bias, model_bias)
        result_bias = catalog_values_encoded_bias.loc[catalog_values_encoded_bias['encoded_target'] == y_result_bias[0]]['target']

        #Train a new model and predict sentiment
        y_encoded_sentiment, catalog_values_encoded_sentiment = encoding_feature(data_sample,'sentiment')
        model_sentiment = train(X,y_encoded_sentiment)
        results_sentiment = evaluate(X, y_encoded_sentiment)
        print(results_sentiment)
        model_sentiment= train(X,y_encoded_sentiment)
        X_new_sentiment = generate_preprocess_new_text(text, count_vectorizer_std)
        y_result_sentiment = pred(X_new_sentiment, model_sentiment)
        result_sentiment = catalog_values_encoded_sentiment.loc[catalog_values_encoded_sentiment['encoded_target'] == y_result_sentiment[0]]['target']

        #Train a new model and predict label
        y_encoded_label, catalog_values_encoded_label = encoding_feature(data_sample,'label')
        model_label= train(X,y_encoded_label)
        results_label = evaluate(X, y_encoded_label)
        print(results_label)
        model_label= train(X,y_encoded_label)
        X_new_label = generate_preprocess_new_text(text, count_vectorizer_std)
        y_result_label = pred(X_new_label, model_label)
        result_label = catalog_values_encoded_label.loc[catalog_values_encoded_label['encoded_target'] == y_result_label[0]]['target']

        print(f"Text: {text}")
        print(f"Bias: {result_bias.item()}")
        print(f"Sentiment: {result_sentiment.item()}")
        print(f"Label: {result_label.item()}")
    except:
        import sys
        import traceback

        #import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
       # ipdb.post_mortem(tb)

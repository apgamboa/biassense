import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import cross_validate
from bias_sense.models.model import naive_bayes, naive_bayes_to_evaluate
from sklearn.naive_bayes import MultinomialNB
from models.model import metrics_NB
from bias_sense.models.gen_ai import text_generator
from dotenv import load_dotenv, find_dotenv
from bias_sense.params import GCP_PROJECT, BUCKET_NAME, ARTIFACTS_FOLDER

from colorama import Fore, Style

from bias_sense.data_layer.data import clean_data, preprocess_features, create_vectorizer, encoding_feature
from bias_sense.data_layer.data import get_data, generate_preprocess_new_text, get_sample_data, get_data_bq
from bias_sense.utils.utilities import load_pickle, save_pickle

# Import upload_all_pickles if it exists in your utilities module
from bias_sense.utils.utilities import upload_all_pickles

load_dotenv(find_dotenv(), override=True)
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")

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


    print("✅ preprocess() done \n")

    return X_processed, count_vectorizer


def train(X:pd.DataFrame,y:np.array):

    """
    -Train dataset
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: train" + Style.RESET_ALL)

    model = naive_bayes(X, y)
    #model.fit(X, y)

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
    metrics_NB(X,y)

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
        text = "This is a text to test in the model a tendency wich policitian, politics and people hates all the time but not religious people"

        path_artifacts = Path(os.path.dirname(__file__))
        path_artifacts = path_artifacts.parent.absolute() / 'artifacts/cloud_training' ## url artifacts
        count_vectorizer_std = load_pickle(path_artifacts, 'count_vectorizer_std.pickle')

        if (count_vectorizer_std is None):
            print("Entra a entrenar de 0 y a guardar modelos")
            #Create data set
            if MODEL_TARGET.lower() == "gcp":
                print("☁️ Cargando datos desde BigQuery")
                data = get_data_bq()
            else:
                print("📂 Cargando datos desde CSV local")
                sdata = get_data()
            data_sample = get_sample_data(data)

            ##Train the model and predict bias
            X, count_vectorizer_std = preprocess(data_sample,'lemmatized_text') # preprocess data and get transformer
            #Get encoded y and catalog encoded
            y_encoded_bias, catalog_values_encoded_bias = encoding_feature(data_sample,'bias_type')
            #Create metrics
           # metrics_bias = evaluate(X, y_encoded_bias)
            more_metrics_bias, model_bias = metrics_NB(X, y_encoded_bias)
            print(more_metrics_bias)
            metrics_bias = pd.DataFrame(more_metrics_bias)
            #Train model
           # model_bias = train(X,y_encoded_bias)


            #Get encoded y and catalog encoded
            y_encoded_sentiment, catalog_values_encoded_sentiment = encoding_feature(data_sample,'sentiment')
            #Evaluate a new model and predict sentiment
            #metrics_sentiment = evaluate(X, y_encoded_sentiment)
            more_metrics_sentiment, model_sentiment = metrics_NB(X, y_encoded_sentiment)
            metrics_sentiment = pd.DataFrame(more_metrics_sentiment)
            #Train a new model and predict sentiment
            #model_sentiment= train(X,y_encoded_sentiment)


            #Train a new model and predict label
            y_encoded_label, catalog_values_encoded_label = encoding_feature(data_sample,'label')
            #Evaluate a new model and predict sentiment
            #metrics_label = evaluate(X, y_encoded_label)
            more_metrics_label, model_label = metrics_NB(X, y_encoded_label)
            metrics_label = pd.DataFrame(more_metrics_label)
            #Train model
            #model_label= train(X,y_encoded_label)

            #Save transformer std
            count_vectorizer_std = save_pickle(path_artifacts, 'count_vectorizer_std.pickle',count_vectorizer_std)
            #Save catalog encoded
            catalog_values_encoded_bias.to_csv(path_artifacts/'catalog_values_encoded_bias.csv', index=False)
            #Save metrics
            metrics_bias.to_csv(path_artifacts /'metrics_bias.csv', index=False)
            #Save model bias
            model_bias = save_pickle(path_artifacts, 'model_bias.pickle',model_bias)

            #Save sentiment objects
            #Save catalog encoded sentiment
            catalog_values_encoded_sentiment.to_csv(path_artifacts/'catalog_values_encoded_sentiment.csv', index=False)
            #Save metrics sentiment
            metrics_sentiment.to_csv(path_artifacts /'metrics_sentiment.csv', index=False)
            #Save model sentiment
            model_sentiment = save_pickle(path_artifacts, 'model_sentiment.pickle',model_sentiment)

            #Save sentiment object
            #Save catalog encoded label
            catalog_values_encoded_label.to_csv(path_artifacts/'catalog_values_encoded_label.csv', index=False)
            #Save metrics label
            metrics_label.to_csv(path_artifacts /'metrics_label.csv', index=False)
            #Save model label
            model_label = save_pickle(path_artifacts, 'model_label.pickle',model_label)
            if MODEL_TARGET.lower() == "gcp":
                print("☁️ Subiendo artefactos a GCS")
                upload_all_pickles()
            print("Termina de entrenar de 0 y de guardar modelos")

        else:
            print("Entra a cargar modelos guardados")
            #load transformer std
            count_vectorizer_std = load_pickle(path_artifacts, 'count_vectorizer_std.pickle')
            #load model bias
            model_bias = load_pickle(path_artifacts, 'model_bias.pickle')
            #load catalog bias
            catalog_values_encoded_bias = pd.read_csv(path_artifacts/'catalog_values_encoded_bias.csv')

            #load model sentiment
            model_sentiment = load_pickle(path_artifacts, 'model_sentiment.pickle')
            #load catalog sentiment
            catalog_values_encoded_sentiment = pd.read_csv(path_artifacts/'catalog_values_encoded_sentiment.csv')

            #load model label
            model_label = load_pickle(path_artifacts, 'model_label.pickle')
            #load catalog label
            catalog_values_encoded_label = pd.read_csv(path_artifacts/'catalog_values_encoded_label.csv')

            print("Termina de a cargar modelos guardados")

        X_new = generate_preprocess_new_text(text, count_vectorizer_std)

        y_result_bias = pred(X_new, model_bias)
        result_bias = catalog_values_encoded_bias.loc[catalog_values_encoded_bias['encoded_target'] == y_result_bias[0]]['target']

        y_result_sentiment = pred(X_new, model_sentiment)
        result_sentiment = catalog_values_encoded_sentiment.loc[catalog_values_encoded_sentiment['encoded_target'] == y_result_sentiment[0]]['target']

        y_result_label = pred(X_new, model_label)
        result_label = catalog_values_encoded_label.loc[catalog_values_encoded_label['encoded_target'] == y_result_label[0]]['target']

        debiased_text = text_generator(text)

        print(f"Text: {text}")
        print(f"Bias: {result_bias.item()}")
        print(f"Sentiment: {result_sentiment.item()}")
        print(f"Label: {result_label.item()}")
        print(f"Debiased text: {debiased_text} ")

    except:
        import sys
        import traceback

        #import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
       # ipdb.post_mortem(tb)

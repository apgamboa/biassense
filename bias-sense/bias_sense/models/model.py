import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def naive_bayes(X:pd.DataFrame,y:np.array) -> MultinomialNB:
    """
    Training with naive bayes model
    """
    naiveBayesModel = MultinomialNB()
    naiveBayesModel.fit(X, y)

    return naiveBayesModel

def naive_bayes_to_evaluate() -> MultinomialNB:
    """
    Get  naive bayes model
    """
    naiveBayesModel = MultinomialNB()

    return naiveBayesModel

def metrics_NB(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 6) # Holdout

    nbm = MultinomialNB()
    nbm.fit(X_train, y_train)
    y_pred = nbm.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred,  pos_label='positive', average='macro'), 2)
    recall = round(recall_score(y_test, y_pred, pos_label='positive', average='macro'), 2)
    f1 = round(f1_score(y_test, y_pred, pos_label='positive', average='macro'), 2)

    dict_metrics ={}
    dict_metrics['accuracy'] = [accuracy]
    dict_metrics['precision'] = [precision]
    dict_metrics['recall'] = [recall]
    dict_metrics['f1'] = [f1]

    return dict_metrics, nbm

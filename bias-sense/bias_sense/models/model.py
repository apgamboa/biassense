import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

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


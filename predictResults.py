import pandas as pd
import numpy as np
from predictClassifier import predictClassifier

def predictResults(xTest, yTest, model, modelTypeFull):

    """
    Predicts new data (as a wrapper for predictClassifier) based on x test data, and saves as csv in current directory.

    Args:

        xTest (pandas dataframe, required): x test data
        yTest (pandas dataframe, required): y test data
        model (model object, required): classifier model object of sklearn
        modelTypeFull (string, required): name of the type of model

    Returns:
        `yPred`: prediction based on x test data
    """

    yPred = predictClassifier(model, xTest)
    yPred_pd = pd.DataFrame(yPred)
    yPred_pd.to_csv('y_prediction_'+modelTypeFull+'.csv')

    np.concatenate((yPred.reshape(len(yPred), 1),
                    yTest.reshape(len(yTest), 1)), 1)

    return yPred

import pandas as pd
import numpy as np
from predictClassifier import predictClassifier

def predictResults(xTest, yTest, model, modelTypeFull):
    yPred = predictClassifier(model, xTest)
    yPred_pd = pd.DataFrame(yPred)
    yPred_pd.to_csv('y_prediction_'+modelTypeFull+'.csv')

    np.concatenate((yPred.reshape(len(yPred), 1),
                    yTest.reshape(len(yTest), 1)), 1)

    return yPred

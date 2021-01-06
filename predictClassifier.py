
def predictClassifier(classifierObj, data):

    """
    Makes Prediction from new data.

    Args:

    classifierObj (model object, required) : model object to be used in predicting new data
    data (dataframe, required): new data to make prediction

    Returns:
        `result`: prediction
    """

    
    result = classifierObj.predict(data)
    return result

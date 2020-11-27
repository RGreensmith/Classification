
def predictClassifier(classifierObj, data):

    """
    Makes Prediction from new data.

    Args:

    classifierObj (model object, required) : model object to be used in predicting new data
    data (dataframe, required): new data to make prediction

    Returns:
    `result`: prediction
    """

    print('predict probability')
    result = classifierObj.predict(data)
    print(result)
    return result

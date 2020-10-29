
def predictClassifier(classifierObj, data):
    print('predict probability')
    result = classifierObj.predict(data)
    print(result)
    return result

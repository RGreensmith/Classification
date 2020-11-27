from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def dataPreprocessor (dataName) :

    """
    Splits the data into x & y training and test, then standardises the data.

    Args:

        dataName (string, required) : filename of the dataset (exclude file extension). Must be a csv, and the y variable must be in the last column.

    Returns:
        `testTrainScalar`: object with 5 attributes - standardised x training data, standardised x test data, y training data, y test data and scalar
    """

    dataset = pd.read_csv(dataName)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    xTrain = sc.fit_transform(xTrain)
    xTest = sc.transform(xTest)

    return xTrain, xTest, yTrain, yTest, sc
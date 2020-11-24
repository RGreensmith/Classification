import pandas as pd
import pickle
from visualiseResults import visualiseResults
from predictResults import predictResults
from plotConfusionMatrix import AccuracyAndConfusion
def classifier (xTrain, yTrain, xTest, yTest, sc, modelType = ('LR', 'KNN', 'SVM', 'KSVM', 'NB', 'DT', 'RF')):
    """
    Creates classifier models

    Args:
        xTrain (pandas dataframe, required): predictor dataframe for model training
        yTrain (pandas dataframe, required): dependent variable dataframe for model training
        xTest  (pandas dataframe, required): predictor dataframe for model testing
        yTest  (pandas dataframe, required): dependent variable dataframe for model testing
        sc (StandardScalar object, required): standard scalar object from sklearn.preprocessing
        modelType (string, optional): classifier models to be created. Defaults to string of all classifiers.
                                        'LR'    = Logistic Regression
                                        'KNN'   = K Nearest Neighbours
                                        'SVM'   = Support Vector Machine
                                        'KSVM'  = Kernel Support Vector Machine
                                        'NB'    = Naive Bayes
                                        'DT'    = Decision Tree
                                        'RF'    = Random Forest

    Returns:
        ``: 
    """
    accuracyScores = [None] * len(modelType)

    for m in range(len(modelType)) :

        if modelType[m] == 'LR' :
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0)
            modelTypeFull = 'Logistic Regression'

        if modelType[m] == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(
                n_neighbors=5, metric='minkowski', p=2)
            modelTypeFull = 'K Nearest Neighbours'

        if modelType[m] == 'SVM' :
            from sklearn.svm import SVC
            model = SVC(kernel='linear', random_state=0)
            modelTypeFull = 'Support Vector Machine'

        if modelType[m] == 'KSVM' :
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', random_state=0)
            modelTypeFull = 'Kernel SVM'

        if modelType[m] == 'NB' :
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            modelTypeFull = 'Naive Bayes'

        if modelType[m] == 'DT' :
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            modelTypeFull = 'Decision Tree'

        if modelType[m] == 'RF' :
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, criterion='entropy', random_state=0)
            modelTypeFull = 'Random Forest'

        model.fit(xTrain, yTrain)

        # save model
        pickle.dump(model, open(modelType[m]+'.sav', 'wb'))

        # Visualise results
        visualiseResults(xTest, yTest, 'test', modelTypeFull, model, sc)
        visualiseResults(xTrain, yTrain, 'train', modelTypeFull, model, sc)

        # Predict Test set results
        yPred = predictResults(xTest, yTest, model, modelTypeFull)

        # confusion matrix and accuracy score
        accuracy = AccuracyAndConfusion(xTest, yTest, yPred, model, modelTypeFull)
        accuracyScores[m] = {"model": modelTypeFull,
                             "accuracyScore": accuracy}
    aS = pd.DataFrame(accuracyScores)
    aS.to_csv('accuracy_Scores.csv')
    

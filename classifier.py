import pandas as pd
import pickle
import time
from visualiseResults import visualiseResults
from predictResults import predictResults
from plotConfusionMatrix import AccuracyAndConfusion
from tabulate import tabulate
def classifier (xTrain, yTrain, xTest, yTest, sc, modelType = ('LR','NB', 'SVM', 'KSVM', 'DT', 'RF', 'KNN')):
    """
    Generates and saves (to directory): classifier models, visualisation plots, accuracy score csv and predicted results csv.

    Args:
        xTrain (pandas dataframe, required): predictor dataframe for model training
        yTrain (pandas dataframe, required): dependent variable dataframe for model training
        xTest  (pandas dataframe, required): predictor dataframe for model testing
        yTest  (pandas dataframe, required): dependent variable dataframe for model testing
        sc (StandardScalar object, required): standard scalar object from sklearn.preprocessing
        modelType (string, optional): classifier models to be created. Defaults to string of all classifiers.
                                        'LR'    = Logistic Regression
                                        'NB'    = Naive Bayes
                                        'SVM'   = Support Vector Machine
                                        'KSVM'  = Kernel Support Vector Machine
                                        'DT'    = Decision Tree
                                        'RF'    = Random Forest
                                        'KNN'   = K Nearest Neighbours

    Returns:
        `Void`
    """
    accuracyScores = [None] * len(modelType)
    times = [None] * len(modelType)

    for m in range(len(modelType)) :
        print(" ")
        print("*****************  STARTING: "+modelType[m])
        if modelType[m] == 'LR' :
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0, verbose=1)
            modelTypeFull = 'Logistic Regression'

        if modelType[m] == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(
                n_neighbors=5, metric='minkowski', p=2)
            modelTypeFull = 'K Nearest Neighbours'

        if modelType[m] == 'SVM' :
            from sklearn.svm import SVC
            model = SVC(kernel='linear', random_state=0, verbose=1)
            modelTypeFull = 'Support Vector Machine'

        if modelType[m] == 'KSVM' :
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', random_state=0, verbose=1)
            modelTypeFull = 'Kernel SVM'

        if modelType[m] == 'NB' :
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            modelTypeFull = 'Naive Bayes'

        if modelType[m] == 'DT' :
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                criterion='entropy', random_state=0)
            modelTypeFull = 'Decision Tree'

        if modelType[m] == 'RF' :
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, criterion='entropy', random_state=0, verbose=1)
            modelTypeFull = 'Random Forest'

        print(" ")
        print("Fitting model")
        print(" ")

        start = time.time()
        model.fit(xTrain, yTrain)
        end = time.time()
        fitTime = "%s" % (end - start)
        print(" ")
        print(modelTypeFull,"fit time: "+fitTime,"seconds")
        del end, start

        # save model
        pickle.dump(model, open(modelType[m]+'.sav', 'wb'))

        print(" ")
        print("Saved model")

        # Visualise results
        print(" ")

        visualiseResults(xTest, yTest, 'test', modelTypeFull, model, sc)
        visualiseResults(xTrain, yTrain, 'train', modelTypeFull, model, sc)

        print(" ")
        print ("Completed result visualisation")

        # Predict Test set results
        start = time.time()
        yPred = predictResults(xTest, yTest, model, modelTypeFull)
        end = time.time()
        predictTime = "%s" % (end - start)
        print(" ")
        print(modelTypeFull,"prediction time: "+predictTime,"seconds")
        del end, start

        # confusion matrix and accuracy score
        accuracy = AccuracyAndConfusion(xTest, yTest, yPred, model, modelTypeFull)
        print(" ")
        print("Saved confusion matrix")

        accuracyScores[m] = {"model": modelTypeFull, "accuracyScore": accuracy}
        times[m] = {"model": modelTypeFull, "fit_seconds": fitTime, "predict_seconds": predictTime}
        
        print(" ")
        print("*****************  Completed: "+modelType[m])
        print(" ")
        if m > 0:
            print(tabulate(accuracyScores[:m+1], headers="keys", floatfmt=".2f"))
            print(" ")
            print(tabulate(times[:m+1], headers="keys", floatfmt=".4f"))
        

    
    aS = pd.DataFrame(accuracyScores)
    T = pd.DataFrame(times)

    aS.to_csv('accuracy_Scores.csv')
    T.to_csv('all_Times.csv')

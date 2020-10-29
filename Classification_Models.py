# Classification
from dataPreprocessor import dataPreprocessor
from classifier import classifier

# Import dataset and preprocess
data = dataPreprocessor('Social_Network_Ads.csv')
xTrain = data[0]
xTest = data[1]
yTrain = data[2]
yTest = data[3]

# Train classification model on the Training set
classifier(xTrain, yTrain, xTest, yTest, sc = data[4])

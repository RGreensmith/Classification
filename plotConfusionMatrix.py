from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def AccuracyAndConfusion(xTest, yTest, yPred, model, modelTypeFull, xLab = 'Age', yLab = 'Salary'):
    """
    Generates and saves a confusion matrix plot to directory, and returns an accuracy score.

    Args:
        xTest (list, required): x test data
        yTest (list, required): y test data
        yPred (list, required): y prediction
        model (model object, required): classifier model object of sklearn
        modelTypeFull (string, required): name of the type of model
        xLab (string, optional): Defaults to 'Age'
        yLab (string, optional): Defaults to 'Salary'

    Returns:
        `accuracy`: accuracy score
    """

    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, xTest, yTest,
                                     display_labels=[xLab, yLab],
                                    cmap=plt.cm.get_cmap('viridis'),
                                    normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.savefig(modelTypeFull+' Confusion Matrix.png')

    cm = confusion_matrix(yTest, yPred)
    print(cm, ' ', modelTypeFull)

    accuracy = accuracy_score(yTest, yPred)
    print(accuracy, ' ', modelTypeFull)
    return accuracy
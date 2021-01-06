from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def AccuracyAndConfusion(xTest, yTest, yPred, model, modelTypeFull):
    """
    Generates and saves a confusion matrix plot to directory, and returns an accuracy score.

    Args:
        xTest (list, required): x test data
        yTest (list, required): y test data
        yPred (list, required): y prediction
        model (model object, required): classifier model object of sklearn
        modelTypeFull (string, required): name of the type of model

    Returns:
        `accuracy`: accuracy score
    """

    classNeg = model.classes_[0]
    classPos = model.classes_[1]

    titles_options = [("Confusion matrix, without normalization", None),
                      (modelTypeFull+ ' Normalized confusion matrix', 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, xTest, yTest,
                                    display_labels=[classNeg, classPos],
                                    cmap=plt.cm.get_cmap('viridis'),
                                    normalize=normalize)
        disp.ax_.set_title(title)

        print(" ")
        print(title)
        print(disp.confusion_matrix)
    plt.savefig(modelTypeFull+' Confusion Matrix.png')
    plt.close()

    cm = confusion_matrix(yTest, yPred)
    print(" ")
    print(modelTypeFull," Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(yTest, yPred)
    print(" ")
    print(modelTypeFull," Accuracy Score:")
    print(accuracy)
    return accuracy

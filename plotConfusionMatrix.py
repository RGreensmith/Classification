from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def AccuracyAndConfusion(xTest, yTest, yPred, model, modelTypeFull, xLab = 'Age', yLab = 'Salary'):
    """[summary]

    Args:
        modelTypeFull ([type]): [description]
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualiseResults(x, y, testOrTrain, modelTypeFull, model, sc, xLab='Age', yLab='Estimated Salary'):

    """
    Generates and saves plots to visualise results to current directory.

    Args:
        x (pandas dataframe, required): data for x axis
        y (pandas dataframe, required): data for y axis
        testOrTrain (string, required): 'test' if using test data, 'train' if using training data.
        modelTypeFull (string, required): name of the type of model.
        model (model object, required): classifier model object of sklearn.
        sc (standardScalar, required): 
        xLab (string, optional): x axis label. Defaults to 'Age'.
        yLab (string, optional): y axis label. Defaults to 'Estimated Salary'.

    Returns:
        `Void`

    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_set, y_set = sc.inverse_transform(x), y
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.6),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.6))
    ax.contourf(x1, x2, model.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                   c=ListedColormap(('red', 'green'))(i), label=j)
    ax.set_title(modelTypeFull+' ('+testOrTrain+' set)')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.legend()
    
    plt.savefig(modelTypeFull+' '+testOrTrain+'.png')
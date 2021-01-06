import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualiseResults(x, y, testOrTrain, modelTypeFull, model, sc, xLab='Age', yLab='Nitrogen Isotope Value'):

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
        yLab (string, optional): y axis label. Defaults to 'Nitrogen Isotope Value'.

    Returns:
        `Void`

    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_set, y_set = sc.inverse_transform(x), y
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.6),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.6))
    
    z = model.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape)
    z2 = np.full_like(z, np.nan, dtype=np.double)

    for a in range(z.shape[0]):
        for b in range(z.shape[1]):
            if z[a, b] == model.classes_[0]:
                z2[a, b] = 0
            else:
                z2[a, b] = 1
    z2 = z2.astype(int)

    ax.contourf(x1, x2, z2,
                alpha=0.75, cmap=ListedColormap(['red', 'blue']))
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                   c=ListedColormap(['red', 'blue'])(i), label=j)
    ax.set_title(modelTypeFull+' ('+testOrTrain+' set)')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.legend()
    
    plt.savefig(modelTypeFull+' '+testOrTrain+'.png')
    plt.close()

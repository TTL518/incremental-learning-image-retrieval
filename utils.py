from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

def create_features_plot(np_features, np_labels):
    '''
    This function create a scatter plot of bidimensional features diffeentiating per class.

    Parameters
    ----------
    features : numpy array
        bx2 matrix of b samples of dimensionality 2.
    labels : numpy array
        bx1 matrix of b labels of dimensionality 1.
    Returns
        PIL image
    -------
    '''

    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    colors = ['blue', 'red', 'orange', 'yellow', 'grey', 'peru', 'cyan', 'violet', 'darkviolet', 'midnightblue']

    '''
    if torch.is_tensor(features):
        np_features = features.cpu().detach().numpy()
    if torch.is_tensor(labels):
        np_labels = labels.cpu().detach().numpy()
    '''
    classes = np.unique(np_labels)

    for i in range(classes.shape[0]):
        indices = np.where(np_labels==classes[i])
        x1 = np_features[indices,0]
        x2 = np_features[indices,1]
        color = colors[classes[i]]
        label = "Class-"+str(classes[i])
        plt.scatter(x1,x2,color=color, marker= '*', label=label)


    # Decorate
    plt.title('Feature class representation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')

    canvas = plt.gcf().canvas

    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    X = np.asarray(agg.buffer_rgba())

    im = Image.fromarray(X)
    return im

    #plt.show()

if __name__ == "__main__":

    feat = torch.rand((10,2))
    labels = torch.randint(0,3,(10,1))

    im = create_features_plot(feat, labels)
    print(im)
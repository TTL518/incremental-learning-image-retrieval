from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.express as px
import numpy as np
import torch
import pandas as pd
import random
import os

def create_features_plot_matplot(np_features, np_labels, class_names):
    '''
    This function create a scatter plot of bidimensional features diffeentiating per class.

    Parameters
    ----------
    np_features : numpy array
        bx2 matrix of b samples of dimensionality 2.
    np_labels : numpy array
        bx1 matrix of b labels of dimensionality 1.
    class_names : dict
        dict with class names associated to class index
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
    classes = np.unique(np_labels, sorted=True)
    #print("CLASSI LEGENDA:", classes)

    fig = plt.figure()

    for i in range(classes.shape[0]):
        indices = np.where(np_labels==classes[i])
        x1 = np_features[indices,0]
        x2 = np_features[indices,1]
        color = colors[classes[i]]
        if(class_names != None):
            label = class_names[i]
        else:
            label = "Class-"+str(classes[i])
        plt.scatter(x1,x2,color=color, marker= '*', label=label)


    # Decorate
    plt.title('Feature class representation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.grid()
    
    return fig

def create_features_plot_plotly(np_features, np_labels, class_names):
    df = pd.DataFrame(np_features, columns = ['Feature1', 'Feature2'])
    df["Class_name"] = [class_names[x.item()] for x in np_labels]
    fig = px.scatter(df, x="Feature1", y="Feature2", color="Class_name")
    return fig

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":

    feat = torch.rand((10,2))
    labels = torch.randint(0,3,(10,1))
    class_names = {
        0:"T-shirt",
        1:"Trousers",
        2:"Pullover",
        3:"Dress",
        4:"Coat",
        5:"Sandal",
        6:"Shirt",
        7:"Sneaker",
        8:"Bag",
        9:"Ankle boot"
    }

    im = create_features_plot_matplot(feat, labels, class_names)
    print(type(im))
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import numpy as np

from itertools import cycle

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# plt.style.use('ggplot')

def load_data(prefix, num_runs):
    spikes_t_arr = []
    spikes_i_arr = []
    I_arr = []
    trace_V_arr = []
    trace_t_arr = []
    labels_arr = []
    for i in range(num_runs):
        spikes_t_arr.append(np.load(prefix+'spikes_t_'+str(i)+'.npy'))
        spikes_i_arr.append(np.load(prefix+'spikes_i_'+str(i)+'.npy'))
        I_arr.append(np.load(prefix+'I_'+str(i)+'.npy'))
        trace_V_arr.append(np.load(prefix+'trace_V_'+str(i)+'.npy'))
        trace_t_arr.append(np.load(prefix+'trace_t_'+str(i)+'.npy'))
        labels_arr.append(np.load(prefix+'labels_'+str(i)+'.npy'))

    return spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, labels_arr

def doPCA(trace_V_arr, k = 3):

    n, num_neurons, length = np.shape(trace_V_arr)

    #concatenate data
    data = np.hstack(trace_V_arr)

    # svd decomposition and extract eigen-values/vectors
    pca = PCA(n_components=k)
    pca.fit(data.T)

    # Save the pca data into each odor/conc
    Xk = pca.transform(data.T)

    pca_arr = []
    for i in range(n):
        pca_arr.append(Xk[length*i:length*(i+1)].T)

    return pca_arr, pca

def applyPCA(PCA, data):
    n, num_neurons, length = np.shape(data)
    data = np.hstack(data)

    Xk = PCA.transform(data.T)

    pca_arr = []
    for i in range(n):
        pca_arr.append(Xk[length*i:length*(i+1)].T)

    return pca_arr


def learnSVM(X, y, K = 'linear'):
    if K == 'linear':
        clf = LinearSVC(C = 0.4, max_iter=100000, verbose = True)
    else:
        clf = SVC(kernel=K, verbose = True, C = 0.6, gamma = 8.0)  
    
    clf.fit(X, y)

    return clf

def normalize(X, minim, maxim):
    return (X-minim)/(maxim - minim)

def plotSVM(clf, X, Y, title, name):
    fig, ax = plt.subplots()

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min()-.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-.1, X[:, 1].max()+.1
    xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))
    Z = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])

    Z = Z.reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25, label = str(Y))
    plt.title(title)
    # plt.legend()

    ax.axis([x_min, x_max,y_min, y_max])

    plt.xticks([])
    plt.yticks([])

    plt.savefig(name, bbox_inches = 'tight')
    plt.show()

def plotPCA2D(pca_arr, title, name, num_trials, skip = 1):
    marker = cycle(['^','o','s','p'])
    cycol = cycle(['y', 'b', 'r'])
    m = next(marker)
    c = next(cycol)

    for i in range(len(pca_arr)):
        d = pca_arr[i]
        m = next(marker)
        if i%num_trials != 0:
            plt.plot(d[0][::skip], d[1][::skip],
             		 '.',
              		 marker = m,
            		 color = c)
        if i%num_trials == 0:
            c = next(cycol)
            marker = cycle(['^','o','s','p'])
            m = next(marker)
            plt.plot(d[0][::skip], d[1][::skip],
                     '.',
                     marker = m,
                     color = c,
                     alpha = 0.4,
                     label = 'base odor '+str((i+num_trials)/num_trials))


    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.title(title, fontsize = 22)

    plt.savefig(name, bbox_inches = 'tight')
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix, 
    
    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()  
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from sklearn.utils import shuffle as rshuffle
from sklearn import metrics
from scipy import stats

import time

from brian2 import *

start_scope()
#path to the data folder
MNIST_data_path = '/Users/Jason/Desktop/Mothnet/MNIST_data/'
# MNIST_data_path = '/home/jplatt/Mothnet/MNIST_data/'

#path to folder to save data
tr_prefix = 'data/'
te_prefix = 'data/test_'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms

numbers_to_inc = frozenset([0, 3])
# numbers_to_inc = frozenset([0, 1, 2])

#size of network 
N_AL = 784 #must be >= 784

num_odors = len(numbers_to_inc)
num_train = 1
num_test = 1
"""
Amount of inhibition between AL Neurons.
Enforces WLC dynamics and needs to be scaled
with the size of the network
"""
in_AL = 0.2

#probability inter-AL connections
PAL = 0.5

input_intensity = 0.2 #scale input
time_per_image = 100 #ms

bin_thresh = 100 #threshold for binary

#--------------------------------------------------------

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo, 
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V']
              )


#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)

net.store()

#-------------------------------------------------------
#load MNIST data
start = time.time()
training = ex.get_labeled_data(MNIST_data_path + 'training', MNIST_data_path)
end = time.time()
print('time needed to load training set:', end - start)

start = time.time()
testing = ex.get_labeled_data(MNIST_data_path + 'testing', MNIST_data_path, bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)

n_input = training['rows']*training['cols'] #28x28=784

num_tot_train = len(training['x'])
imgs_train = training['x']
labels_train = training['y']

num_tot_test = len(testing['x'])
imgs_test = testing['x']
labels_test = testing['y']


imgs_train, labels_train = rshuffle(imgs_train, labels_train)
imgs_test, labels_test = rshuffle(imgs_test, labels_test)


#-------------------------------------------------------

run_params_train = dict(num_trials = num_train,
                        prefix = tr_prefix,
                        input_intensity = input_intensity,
                        time_per_image = time_per_image,
                        N_AL = N_AL,
                        labels = labels_train,
                        numbers_to_inc = numbers_to_inc,
                        bin_thresh = bin_thresh,
                        n_input = n_input,
                        )


states = dict(  G_AL = G_AL,
                S_AL = S_AL,
                trace_AL = trace_AL,
                spikes_AL = spikes_AL)
 

run_params_test = dict( num_trials = num_test,
                        prefix = te_prefix,
                        input_intensity = input_intensity,
                        time_per_image = time_per_image,
                        N_AL = N_AL,
                        labels = labels_test,
                        numbers_to_inc = numbers_to_inc,
                        bin_thresh = bin_thresh,
                        n_input = n_input,
                        )

ex.runMNIST(run_params_train, imgs_train, states, net)
ex.runMNIST(run_params_test, imgs_test, states, net)

#---------------------------------------------------------

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)


skip = 4
#uncomment these lines to do PCA on the output
pca_dim = 2
pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

X = np.hstack(pca_arr).T
# X = np.hstack(trace_V_arr).T


mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)
y = np.hstack(label_arr)

X = X[::skip]
y = y[::skip]

clf = anal.learnSVM(X, y, K = 'linear')

#--------------------------------------------------------

test_data = anal.applyPCA(PCA, test_V_arr)
# test_data = test_V_arr
test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(label_test_arr, axis = 1)

pred_arr = []
for i in range(len(test_data)):
    pred = clf.predict(test_data[i].T)
    total_pred = stats.mode(pred)[0]
    print('True: ' + str(y_test[i]), 'pred: ' + str(int(total_pred)))
    pred_arr.append(total_pred)

expected = y_test
predicted = np.array(pred_arr)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)


print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

title = 'Training Data Boundary'
name = 'boundary.pdf'
anal.plotSVM(clf, X, y, title, name)
title = 'Testing MNIST'
name = 'testing.pdf'
anal.plotSVM(clf, test_data, label_test_arr, title, name)


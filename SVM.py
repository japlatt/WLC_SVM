from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats

import matplotlib.pyplot as plt
from sklearn import metrics

defaultclock.dt = .05*ms

np.random.seed(125)

N_AL = 1000
in_AL = .1
PAL = 0.5

tr_prefix = 'data_dev/'
te_prefix = 'data_dev/test_'

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

inp = 0.15
noise_amp = 0.0 #max noise percentage of inp
noise_test = 2.0*np.sqrt(3)

num_odors = 2

num_train = 1

num_test = 1

run_time = 120*ms

I_arr = []
#create the base odors
for i in range(num_odors):
    I = ex.get_rand_I(N_AL, p = 0.33, I_inp = inp)*nA
    I_arr.append(I)

run_params_train = dict(num_odors = num_odors,
                        num_trials = num_train,
                        prefix = tr_prefix,
                        inp = inp,
                        noise_amp = noise_amp,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = True)


states = dict(  G_AL = G_AL,
                S_AL = S_AL,
                trace_AL = trace_AL,
                spikes_AL = spikes_AL)
 

run_params_test = dict( num_odors = num_odors,
                        num_trials = num_test,
                        prefix = te_prefix,
                        inp = inp,
                        noise_amp = noise_test,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = False)

ex.createData(run_params_train, I_arr, states, net)
ex.createData(run_params_test, I_arr, states, net)

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)


#uncomment these lines to do PCA on the output
pca_dim = 2
pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

X = np.hstack(pca_arr).T
# X = np.hstack(trace_V_arr).T


mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)

pca_arr = anal.normalize(pca_arr, mini, maxi)

y = np.hstack(label_arr)

clf = anal.learnSVM(X, y)

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


# Only works if pca_dim = 2


# title = 'Arbitrary Input Training'
# name = 'training.pdf'
# anal.plotPCA2D(pca_arr, title, name, num_train, skip = 2)
title = 'Training Boundary'
name = tr_prefix+'boundary_AI.pdf'
anal.plotSVM(clf, X, y, title, name)
title = 'Testing Input with Noise ' + str(np.rint(100*noise_test/sqrt(3)))+'%'
name = tr_prefix+'testing_AI.pdf'
anal.plotSVM(clf, test_data, label_test_arr, title, name)

#save text files for sending to Henry
c = 0
for i in range(num_odors):
    for j in range(num_train):
        np.savetxt(tr_prefix+'Train_SVM2D_%d%d.dat'%(i, j),
                   np.append(pca_arr[c].T, np.reshape(label_arr[c], (-1,1)), 1),
                   fmt = '%1.3f') 
        np.savetxt(tr_prefix+'Test_SVM2D_%d%d.dat'%(i, j),
                   np.append(test_data[c].T, np.reshape(clf.predict(test_data[c].T), (-1,1)), 1),
                   fmt = '%1.3f') 
        c = c+1





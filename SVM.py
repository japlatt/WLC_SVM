from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal

import matplotlib.pyplot as plt
from sklearn import metrics

defaultclock.dt = .05*ms

np.random.seed(22)

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
noise_amp = 0.1 #max noise percentage of inp
noise_test = 0.8

num_odors = 10

num_train = 1

num_test = 2

run_time = 80*ms

I_arr = []
#create the base odors
for i in range(num_odors):
    I = ex.get_rand_I(N_AL, p = np.random.uniform(0.1, 0.5), I_inp = inp)*nA
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

# ex.createData(run_params_train, I_arr, states, net)
# ex.createData(run_params_test, I_arr, states, net)

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)


pca_dim = 20
pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

print(pca_arr[0])

X = np.hstack(pca_arr).T
mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)
y = np.hstack(label_arr)

clf = anal.learnSVM(X, y)

test_data = anal.applyPCA(PCA, test_V_arr)
test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(label_test_arr, axis = 1)

pred_arr = []
for i in range(len(test_data)):
    pred = clf.predict(test_data[i].T)
    total_pred = np.rint(np.mean(pred))
    print('True: ' + str(y_test[i]), 'pred: ' + str(int(total_pred)))
    pred_arr.append(total_pred)

expected = y_test
predicted = np.array(pred_arr)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

# anal.plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

# title = 'Arbitrary Input Training'
# name = 'training.pdf'
# anal.plotPCA2D(pca_arr, title, name, num_train, skip = 2)
# title = 'Arbitrary Input Training Boundary'
# name = 'boundary_AI.pdf'
# anal.plotSVM(clf, X, y, title, name)
# title = 'Testing Arbitrary Input with Noise ' + str(np.rint(100*noise_test/sqrt(3)))+'%'
# name = 'testing_AI.pdf'
# anal.plotSVM(clf, test_data, label_test_arr, title, name)





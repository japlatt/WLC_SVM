from __future__ import division

from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

defaultclock.dt = .05*ms

np.random.seed(125)

N_AL = 1000 #number of neurons in network
in_AL = .1 #inhibition g_nt in paper
PAL = 0.5 #probability of connection

#folders in which to save training and testing data
tr_prefix = 'train/'
te_prefix = 'test/test_'

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

inp = 0.15 #input current amplitude
noise_amp = 0.0 #max noise percentage of inp
noise_test = 0.0 #no noise in the mixtures

num_odors_train = 10 #how many buckets
num_odors_mix = 2 #mix 2 of the odors together

num_alpha = 100 #values of A in: A*I_1 + (1-A)*I_2

num_test = 1 #test per value of A

run_time = 120*ms

I_arr = []
#create the base odors
for i in range(num_odors_train):
    I = ex.get_rand_I(N_AL, p = 0.33, I_inp = inp)*nA
    I_arr.append(I)

run_params_train = dict(num_odors = num_odors_train,
                        num_trials = 1,
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
 

run_params_test = dict( num_odors = num_odors_mix,
                        num_trials = num_alpha,
                        prefix = te_prefix,
                        inp = inp,
                        noise_amp = noise_test,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = False)

# --------------------------------------------------------------
#run the simulation and save to disk
ex.createData(run_params_train, I_arr, states, net)
ex.mixtures2(run_params_test, I_arr[:num_odors_mix], states, net)

#load in the data from disk
spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_alpha*num_test)

X = np.hstack(trace_V_arr).T

#normalize training data
mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)

y = np.hstack(label_arr)

#train the SVM
clf = anal.learnSVM(X, y)

test_data = test_V_arr
test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(label_test_arr, axis = 1)

pred_arr = []
A_arr = []
for i in range(len(test_data)):
    pred = clf.predict(test_data[i].T)
    total_pred = stats.mode(pred)[0]
    pred_arr.append(total_pred)
    A_arr.append(np.histogram(pred, bins = np.arange(num_odors_train+1))[0])

A_arr = np.array(A_arr)/np.sum(A_arr[0]) #alpha array

np.savetxt(tr_prefix+'alpha_histogram.txt', A_arr, fmt = '%1.3f')
expected = y_test
predicted = np.array(pred_arr)

x = A_arr[:, 0]
y = A_arr[:, 1]
z = A_arr[:, 2]

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

A = np.arange(num_alpha)/num_alpha

popt_x, pcov_x = curve_fit(fsigmoid, A, x)
popt_y, pcov_y = curve_fit(fsigmoid, A, y)

print popt_x
print popt_y
#----------------------------------------------------
#Plotting

plt.plot(A, y, 'r.', label = r'$P(I_1)$')
plt.plot(A, fsigmoid(A, *popt_y), 'r-', linewidth = 2)
plt.plot(A, x, 'b.', label = r'$P(I_2)$')
plt.plot(A, fsigmoid(A, *popt_x), 'b-', linewidth = 2)
plt.plot(A, z, 'k.', label = r'1-$P(I_1)$-$P(I_2)$')
plt.title('Classification of the Mixture of 2 Odors', fontsize = 20)
plt.xlabel(r'$\alpha$ in $\alpha I_1 + (1-\alpha) I_2$', fontsize = 16)
plt.ylabel(r'Classification Probability of $I_1/I_2$', fontsize = 16)
plt.legend()

plt.show()

plt.savefig('mixtures.pdf', bbox = 'tight')



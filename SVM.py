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

net.store() #save the connected network

inp = 0.15 #input current amplitude
noise_amp = 0.0 #max noise percentage of inp
noise_test = 2.0/np.sqrt(3) #NSR = 1/SNR = noise_test*sqrt(3)

num_odors = 2 #total number of odors (classes)

num_train = 1 #number of training odors per odor in num_odors

num_test = 2 #number of testing odors per odor in num_odors

run_time = 120*ms #run time per presentation

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

#create the data and save it to disk
ex.createData(run_params_train, I_arr, states, net)
ex.createData(run_params_test, I_arr, states, net)

#load in the data from disk
spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)

#-------------------------------------------------------------------
#SVM

pca = True #do PCA on output or not

if pca: #do PCA on the output
    pca_dim = 2
    pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

    X = np.hstack(pca_arr).T
else:
    X = np.hstack(trace_V_arr).T


#normalize
mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)

y = np.hstack(label_arr)

#train the SVM
clf = anal.learnSVM(X, y)

#--------------------------------------------------------------
#test SVM
if pca:
    test_data = anal.applyPCA(PCA, test_V_arr)
else:
    test_data = test_V_arr

test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(label_test_arr, axis = 1)

#check predictions on the test data
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

#----------------------------------------------------------------------
#Plotting

# Plotting only works if pca_dim = 2 or 3
if pca and pca_dim == 2:
    #plot no training boundary
    title = 'PCA 2D'
    name = tr_prefix+'PCA_2D.pdf'
    anal.plotPCA2D(pca_arr, title, name, num_train, skip = 2)

    #plot the training boundary
    title = 'Training Boundary 2D'
    name = tr_prefix+'boundary_AI.pdf'
    anal.plotSVM(clf, X, y, title, name)

    #plot the test data
    title = 'Testing Input with Noise ' + str(np.rint(100*noise_test/sqrt(3)))+'%'
    name = tr_prefix+'testing_AI.pdf'
    anal.plotSVM(clf, test_data, label_test_arr, title, name)


if pca and pca_dim == 3:
    title = 'PCA 3D'
    name = tr_prefix+'PCA_3D.pdf'
    anal.plotPCA3D(pca_arr, N_AL, title, name, el = 30, az = 30, skip = 1, start = 0)


#-------------------------------------------------------------
see_InCA = False #set to false to skip InCA
get_mim = True #set to false if computed already

# uncomment these lines to do InCA on the output

if see_InCA:
    if get_mim: anal.getMIM(tr_prefix, trace_V_arr) #takes a long time
    MIM = np.load(tr_prefix+'MIM.npy')


    InCA_dim = 2
    InCAdata = anal.doInCA(MIM, trace_V_arr, skip = 1, k = InCA_dim)
    title = 'HH InCA Projection'

    if InCA_dim == 2: 
        name = 'HH_InCA_2D.pdf'
        anal.plotPCA2D(InCAdata, title, name, num_train, skip = 1)
    if InCA_dim == 3: 
        name = 'HH_InCA_3D.pdf'
        anal.plotPCA3D(InCAdata, N_AL, title, name, el = 0, az = 0, skip = 1, start = 0)


# np.save(tr_prefix+'HH_InCA_%dD.npy' %InCA_dim, InCAdata)

# plt.show()



# save text files
# c = 0
# for i in range(num_odors):
#     for j in range(num_train):
#         np.savetxt(tr_prefix+'Train_SVM2D_%d%d.dat'%(i, j),
#                    np.append(pca_arr[c].T, np.reshape(label_arr[c], (-1,1)), 1),
#                    fmt = '%1.3f') 
#         np.savetxt(tr_prefix+'Test_SVM2D_%d%d.dat'%(i, j),
#                    np.append(test_data[c].T, np.reshape(clf.predict(test_data[c].T), (-1,1)), 1),
#                    fmt = '%1.3f') 
#         c = c+1





import numpy as np
import os.path
from struct import unpack
import cPickle as pickle
import brian2.only as br
from sklearn.utils import shuffle as rshuffle


def createData(run_params, I_arr, states, net):
    num_odors = run_params['num_odors']
    num_trials = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['inp']
    noise_amp = run_params['noise_amp']
    run_time = run_params['run_time']
    N_AL = run_params['N_AL']
    train = run_params['train']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']

    n = 0 #Counting index
    start = 100 #skip the first start*dt ms to remove transients
    for j in range(num_odors):
        for k in range(num_trials):
            noise = noise_amp#*np.random.randn()
            net.restore()

            if k == 0 and train:
                G_AL.I_inj = I_arr[j]
            else:
                G_AL.I_inj = I_arr[j]+noise*inp*(2*np.random.random(N_AL)-1)*br.nA

            net.run(run_time, report = 'text')

            np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n), G_AL.I_inj)
            np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
            np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])

            np.save(prefix+'labels_'+str(n), np.ones(len(trace_AL.t[start:]))*j)
            n = n+1


def runMNIST(run_params, imgs, states, net):
    num_train = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['input_intensity']
    run_time = run_params['time_per_image']
    N_AL = run_params['N_AL']
    labels = run_params['labels']
    numbers_to_inc = run_params['numbers_to_inc']
    bin_thresh = run_params['bin_thresh']
    n_input = run_params['n_input']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']

    num_run = np.zeros(len(numbers_to_inc))

    start = 330
    #run the network
    n = 0
    for i in range(60000):
        y = labels[i][0]
        if y in numbers_to_inc and num_run[y] < num_train:
            net.restore()
            print('image: ' + str(n))
            print('label: '+ str(labels[i][0]))
            
            #right now creating binary image
            rates = rates = np.where(imgs[i%60000,:,:] > bin_thresh, 1, 0)*inp

            linear = np.ravel(rates)
            padding = N_AL - n_input
            I = np.pad(linear, (0,padding), 'constant', constant_values=(0,0))
            I = I+rshuffle(I, random_state = 10)+rshuffle(I, random_state = 2)

            I = np.where(I > inp, inp, I)
            print(np.sum(I/inp))

            G_AL.I_inj = br.nA*I
            
            net.run(run_time*br.ms, report = 'text')

            np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n), I)
            np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
            np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])

            np.save(prefix+'labels_'+str(n), np.ones(len(trace_AL.t[start:]))*y)
            n = n+1
            num_run[y] = num_run[y]+1
        if np.all(num_run == num_train):
            break
'''
Random constant current input
N: size of array
p: probability of index having input
I_inp: strength of current
'''
def get_rand_I(N, p, I_inp):
    I = np.random.random(N)
    I[I > p] = 0
    I[np.nonzero(I)] = I_inp
    return I


'''
MNIST helper function to load and unpack MNIST data.  To run you need to download 
the MNIST dataset from http://yann.lecun.com/exdb/mnist/.

picklename: name of file to write/read data from
MNIST_data_path: path to the MNIST data folder
bTrain: if tr
'''
def get_labeled_data(picklename, MNIST_data_path, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))#, encoding='utf-8')
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"), -1)
    return data
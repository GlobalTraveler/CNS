#from pylab import *
import numpy as np
from pylab import *
import scipy.linalg

# np.random.seed(0)
class bm(object):
    def __init__(self, data, noiselevel = .4):
        images, targets = data

        # binarize + add noise; do this as a function of noise levels
        noise = np.random.rand(images.shape[0], images.shape[1])
        images[images > 0] = 1
        images[images ==0] = -1

        # # adding noise
        x, y = np.where(noise < noiselevel)
        noise = np.random.randint(0, 2, (x.shape)) * 2 - 1
        images[x, y] = noise


        self.n_neurons = images.shape[1]
        self.clampedMean = np.array(np.mean(images, axis = 0), ndmin = 2)
        self.clampedCov = images.T.dot(images) / len(images)
        self.weights, self.theta, self.F = self.meanField()

    def meanField(self):
        '''
        Computes meanfield approximation
        '''

        n_neurons = self.n_neurons
        # compute clamped statisticss
        mi = self.clampedMean.T.copy()
        chi = self.clampedCov - self.clampedMean.flatten() * self.clampedMean.flatten()
        # invert: noise is added to the data to prevent singularity issues
        invC = np.linalg.inv(chi)
        # invC = np.linalg.solve(chi, np.eye(n_neurons))

        weights = np.diag(1/(1 - mi.flatten()**2)) - invC
        # m contains both the theta and the weight.T.dot(m), so in order to extract
        # theta we :
        theta = np.arctanh(mi) - weights.dot(mi)
        # theta = np.arctan(mi) - weights.dot(mi)
        F = - .5 * ( weights.dot(mi).T.dot(mi)) - theta.T.dot(mi) + \
        .5 * (\
            (1 + mi).T.dot( np.log(.5 *  ( 1 + mi )) )+ (1 - mi ).T.dot(np.log(.5 * (1 - mi)))\
            )
        return weights, theta, F

def classify(machines, images, targets):
    '''
    This function classifies the input images and computes the performance
    of the classifiers
    '''
    ps = []
    assignment = np.zeros(len(targets))
    correct = np.zeros(assignment.shape)
    print('starting classification')
    for si, s in enumerate(images):
        p = np.zeros(len(machines))
        for idx, machine in enumerate(machines):
            # extract correct values [easier typing]
            weights = machine.weights
            theta = machine.theta
            # log probability log(z) = -F ==> -log(z) = F
            p[idx] = machine.F + .5 * s.T.dot(weights).dot(s) + theta.flatten().dot(s)
        assignment[si] = int(np.argmax(p))
        correct[si] = assignment[si] == targets[si]
    return assignment, np.mean(correct)

def experiment(noiselevel):
    machines = []
    for i in range(10):
        # obtain the indices corresponding to  a number and train a bm on it
        idx = train_targets == i
        machines.append(bm(data = [train_images[idx,:], train_targets[idx]], noiselevel = noiselevel))
    ass, corr = classify(machines, test_images, test_targets)
    print(corr)
    # create confusion matrix
    confusion = np.zeros((10,10))
    # loop through the targets and see whether the assignment matches
    for idx, target in enumerate(test_targets):
        confusion[target, int(ass[idx])] += 1
    return ass, corr, confusion


import os
print(os.getcwd())
# from h5py import File
#
# with File('mnist_data/train-labels-idx1-ubyte') as f:
#     for i in f: print(i)
from mnist import MNIST as m
# folder to the file
load = m('mnist_data/')
shape_im = (28,28)

# load the data
train_images, train_targets = load.load_training()
test_images, test_targets   = load.load_testing()

# convert to single precision
test_images                 = np.array(test_images, dtype = np.float32)

train_images                = np.array(train_images, dtype = np.float32)

# convert to numpy arrays [easy reshaping]
test_targets                = np.array(test_targets)
train_targets               = np.array(train_targets)

# binarize images
test_images[test_images >  0] = 1
test_images[test_images == 0] = -1

train_images[train_images > 0] = 1
train_images[train_images == 0] = -1

noises = linspace(.1,1,10)
results = []
for idx, noise in enumerate(noises):
    results.append(experiment(noise))
results = np.array(results)

# plot the performance as function of noise level
fig, ax = subplots()
ax.plot(noises, results[:,1])
ax.set_xlabel('Noise level')
ax.set_ylabel('Perfrmance')

fig = figure()
for idx, data in enumerate(results):
    ax = fig.add_subplot(5,2,idx+1)
    ax.imshow(data[-1])
    ax.set_title('Noise level = {0:.1f}'.format(noises[idx]))
    ax.set_ylabel('Number')
    ax.set_xlabel('Number')
# fig.tight_layout()
show()

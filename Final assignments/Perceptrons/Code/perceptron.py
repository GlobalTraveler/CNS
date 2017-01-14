import numpy as np
from scipy.io import loadmat
from h5py import File
from pylab import *

'''
To do:
    normal perceptron (gradient descent?)
    stochastic perceptron
    non-linear activation
    multi-layer perceptron
issues: what activation function to use?
'''

def data_loader(file):

    # we only need the labels of 7 and 3
    with File(file) as f:
        group = f['mnist']['train_labels']
        images = f['mnist']['train_images']
        targets = [7, 3]
        train_labels = []
        train_images = []
        for idx, t in enumerate(targets):
            # only 1D target vector so only look at the columns
            tmp = np.where(group.value == t)[1]
            tmp_images = images.value[tmp, :]

            # store the train images and target labels
            train_labels.append(group.value[0,tmp])

            train_images.append(tmp_images)


    # plot example train_images

    # fig, ax = subplots(1,2)
    # ax[0].imshow(train_images[0][0,:].T)
    # ax[1].imshow(train_images[1][0,:].T)

    # stack the images
    train_images = np.vstack((train_images[0], train_images[1]))


    # reshape 28x28 = 748
    train_images = \
    train_images.reshape(\
    train_images.shape[0], \
    train_images.shape[1]**2)

    # stack the targets
    train_targets = np.hstack((train_labels[0], train_labels[1])).astype(np.int)


    # print(train_targets.dtype)
    # binarize targets
    train_targets[train_targets == targets[0]] = -1
    train_targets[train_targets == targets[1]] = 1

    # binarize images
    train_images[train_images > 0] = 1


    # print(np.unique(train_targets))
    find_7 = np.where(train_targets == -1)[0][0]
    find_3 = np.where(train_targets == 1)[0][0]

    # show binarized sample images
    # fig, ax = subplots(1,2)
    # ax[0].imshow(train_images[find_3,:].reshape(28,28).T)
    # ax[1].imshow(train_images[find_7,:].reshape(28,28).T)

    return train_images, train_targets


class perceptron(object):
    def __init__(self,\
                 file = '../Data/mnist_all.mat',\
                 moment_batch = None,\
                 eta = .1):

        self.train_images, self.train_targets = data_loader(file)
        print(self.train_targets.shape)

        # moment batch contains a config file
        if moment_batch != None:
            self.moment_batch, self.gamma = moment_batch
        else:
            self.moment_batch   = 0
            self.gamma          = 0
        self.eta = eta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def batch_learning(self, data, targets):
        # append bias term
        data = np.hstack((data, np.ones((data.shape[0], 1))))

        # init weights
        weights = np.random.rand(data.shape[1]) * 2 - 1


        if self.moment_batch:
            momentum = np.zeros(weights.shape)
        iters = 0
        # sum squared error
        MSE = []

        # keep looping until smaller than machine error
        alpha = True
        while alpha:
            output = self.sigmoid(data.dot(weights))
            # output = data.dot(weights)
            # print(np.unique(output))
            error = targets - output


            if self.moment_batch:
                delta_weights   =   self.eta * data.T.dot(error)
                weights         +=  self.gamma  * momentum + delta_weights
                momentum        =   delta_weights
            else:
                weights += self.eta * data.T.dot(error)

            MSE.append( .5  * np.mean(error**2) )
                # print(SSE[-1])

            if MSE[-1] == inf:
                alpha = False

            some_range = 100
            if iters > some_range * 2 :
                value = np.sum(abs(np.diff(MSE[-some_range:])))
                if value < np.finfo(np.float).eps:
#                    print(value)
                    alpha = False
            # print(iters)
            if iters > 10000:
                alpha = False
            iters += 1
            # print(iters)
        self.batch_weights = weights
        return MSE, iters


    def stochastic_learning(self):
        return 0

1e24
momentum_batch = [1, 1]
momentum_batch = None
#momentum_batch = None
per = perceptron(eta = 1e-6, moment_batch = momentum_batch)
MSE, iters = per.batch_learning(per.train_images, per.train_targets)

print(iters, np.min(MSE))
fig, ax =  subplots(1,1)
ax.plot(MSE)
show(block = 1)
close('all')

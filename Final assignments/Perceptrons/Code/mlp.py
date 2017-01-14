import numpy as np
from pylab import *
from h5py import File


def mlp(X, t, eta = 1e-1, gamma = 0, num = 1e3, K = 1, M = 8, save = 4):
    '''
    Multi-layered-pereptron
    K = output nodes
    M = hidden nodes
    Assumes the input data X is samples x feature dimension
    Returns:
        prediction and error
    '''
    num = int(num)
    # add bias node; note the bias is absorbed in the weights of the
    # second layer ()
    M = M #+ 1
    # input dimensi            a = input.dot(weights)
    D = X.shape[1] + 1
    # stack bias constant signal of ones
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    # init weights
    # hidden weights and outputweights
    wh = np.random.rand(D, M) - 1/2
    wo = np.random.rand(M, K) - 1/2
    errors = np.zeros(num + 1)
    preds = [[]] * (save + 1)
    idx = 0
    tmp = num // save
    for i in range(num + 1):
        # forward pass
        a = X.dot(wh);  z = np.tanh(a);   y = z.dot(wo)
        # backward pass
        dk = t - y
        # compute hidden activation; note elementwise product!!
        dj = (1 - z**2) * ((wo.dot(dk.T).T))
        # dj = z * (1 - z ) * ((wo.dot(dk.T)).T)
        # update th e weights
        E1 = z.T.dot(dk); E2 = X.T.dot(dj)
        wo += eta * E1 + gamma * wo
        wh += eta * E2 + gamma * wh
        error = np.sum((y-t)**2)
        errors[i] = error
        # print(error)
        if error < 1e-1:
            print('error low enough')
            break
        if error == nan:
            print('nan encountered')
            break
        if  i % tmp == 0:
            print('pass', i)
            preds[idx] = [y,i]
            idx += 1
        # print(i)
    return errors, preds


fileDir = '../Data/mnist_all.mat'

with File(fileDir) as f:
    # plot directory overview
    for i in f['mnist']: print(i)
    g = f['mnist']
    dataLabels = ['train_labels', 'train_images', 'test_labels', 'test_images']
    data = []
    for label in dataLabels:
        tmp = g[label].value
        if len(tmp.shape) > 2:
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]**2)
            tmp = tmp[interesting , : ]

        else:
            threes = np.where(tmp == 3)[1]
            sevens = np.where(tmp == 7)[1]
            interesting = np.hstack((threes,sevens))
            tmp = tmp[:, interesting].T
            print(tmp.shape)
        data.append(tmp)

trainTargets, trainImages, testTargets, testImages = data
errors, preds = mlp(trainImages, trainTargets, eta = 1e-5, num = 800)

fig, ax  = subplots()
ax.plot(errors)
show()

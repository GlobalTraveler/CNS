class mlp(object):
    '''
    Multi-layered-pereptron
    K = output nodes
    M = hidden nodes
    Assumes the input data X is samples x feature dimension
    Returns:
        prediction and error
    '''
    def __init__(self, X, t,\
                 eta = 1e-1,\
                 gamma = .0,\
                 M = 8,\
                 K = 1):
        # learning rate / momentum rate
        self.eta        = eta
        self.gamma      = gamma
        # Layer dimensions; input, hidden, output
        self.D          = D =  X.shape[1] + 1
        self.M          = M
        self.K          = K
        # add bias node to input
        self.X          = hstack( (X, ones(( X.shape[0], 1 ) ) ) )
        self.targets    = t
        # weights; hidden and output
        wh              = random.rand(D, M) - 1/2
        wo              = random.rand(M, K) - 1/2

        self.layers     = [wh, wo]
        # activation functions:
        self.func       = lambda x: tanh(x)
        self.dfunc      = lambda x: 1 - x**2


    def forwardSingle(self, xi):
        ''' Performs a single forward pass in the network'''
        layerOutputs = [ []  for j in self.layers ]
        #forward pass
        a = xi.dot(self.layers[0])
        z = self.func(a)
        y = z.dot(self.layers[1])

        # save output
        layerOutputs[0].append(z);
        layerOutputs[1].append(y)
        return layerOutputs

    def backwardsSingle(self, ti, xi, forwardPass):
        '''Backprop + update of weights'''
        # prediction error
        dk = forwardPass[-1][0] - ti
        squaredError = dk**2
        # compute hidden activation; note elementwise product!!
        dj = \
        self.dfunc(forwardPass[0][0]) * (dk.dot(self.layers[-1].T))

        # update the weights
        E1 = forwardPass[0][0].T.dot(dk)
        E2 = xi.T.dot(dj)

        # update weights of layers
        self.layers[-1] -= \
        self.eta * E1 + self.gamma * self.layers[-1]

        self.layers[0]  -= \
        self.eta * E2 + self.gamma * self.layers[0]
        return squaredError

    def train(self, num, funcs = [self.batch], plotProg = (False,)):
        #set up figure
        if plotProg[0]:
            fig, ax = subplots(subplot_kw = {'projection':'3d'})


        num   = int(num) # for scientific notation
        SSE   = zeros(num) # sum squared error
        preds = zeros((num, len(self.targets))) # predictions per run
        for iter in range(num):
            error = 0 # sum squared error
            for idx, (ti, xi) in enumerate(zip(self.targets, self.X)):
                xi = array(xi, ndmin = 2)

                forwardPass = self.forwardSingle(xi)
                error += self.backwardsSingle(ti, xi, forwardPass)
                preds[iter, idx] = forwardPass[-1][0]
            # plot progress
            if plotProg[0]:
                if not iter % plotProg[1]:
                    x, y = plotProg[2]
                    ax.cla() # ugly workaround
                    ax.plot_surface(x, y, preds[iter, :].reshape(x.shape))
                    ax.set_xlabel('$x_1$', labelpad = 20)
                    ax.set_ylabel('$x_2$', labelpad = 20)
                    ax.set_zlabel('pdf', labelpad =20)
                    ax.set_title('Cycle = {0}'.format( iter ))
                    pause(1e-10)
            SSE[iter] = .5 * error
        return SSE, preds




import numpy as np
from pylab import *
from h5py import File




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

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
                 eta = 1e-5,\
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
        wh              = random.randn(D, M)
        wo              = random.randn(M, K)

        self.layers     = [wh, wo]
        self.nLayers    = len(self.layers)
        # activation functions:
        self.func       = lambda x: tanh(x)
        self.dfunc      = lambda x: 1 - x**2
        # cost function
        self.cost       = lambda x: sum(x**2)/len(self.targets) #MSE

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
    def conjgrad(self, fw, tol = 1e-2):
        t = self.targets
        fw = self.func(X.dot(self.layers[0])).dot(self.layers[-1])
        p = r = t - fw
        rsold = r.T.dot(r)
        for ti in t:
            Ap = self.layers[-1].dot(p)
            alpha = rsold / (p.T.dot(Ap))

    def updateForward(self, X):
        '''Returns the gradients and activation'''
        output = X
        # storage for activation / gradient per layer
        update = empty(( len(self.layers) + 1, 2), dtype = object )
        for idx, _ in enumerate(update):
            if idx == 0:
                update[idx, 0] = output
            else:
                layer           = self.layers[idx - 1]
                output          = self.func(output.dot(layer))
                gradient        = self.dfunc(output)
                update[idx, :] = [output, gradient]
        # print(max(output))
        return update

    def updateBack(self, t, forwardPass):
        # forward pass is one index longer, the first index contains the input
        for idx in range(self.nLayers - 1, -1, -1):
            prediction, gradient = forwardPass[idx + 1]
            if idx == self.nLayers - 1 :
                error = t - prediction
                # print(sum(error))
                saveError = error
            else:
                error = gradient * ( self.layers[idx + 1].dot(error.T) ).T
            # print(forwardPass[idx][0].shape); assert 0
            update = forwardPass[idx][0].T.dot(error)
            # print(idx, error.shape, update.shape, self.layers[idx].shape)
            self.layers[idx] += self.eta * update + self.gamma * self.layers[idx]
        return self.cost(saveError)

    def train(self, num):
        tmp = []
        for i in range(num):
            f = self.updateForward(self.X)
            tmp.append(self.updateBack(self.targets, f))
        return tmp




    # def train(self, num, plotProg = (False,)):
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

            # plot progressnum = 100
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
            if SSE[iter] < 1e-3:
                print('Error is sufficiently low')
                break
        return SSE, preds

from pylab import *
from numpy import *
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
            tmp = np.array(tmp[interesting , : ], dtype = int)

        else:
            threes = np.where(tmp == 3)[1]
            sevens = np.where(tmp == 7)[1]
            interesting = np.hstack((threes,sevens))
            tmp = np.array(tmp[:, interesting].T, dtype = int)
            # print(tmp.shape)
        data.append(tmp)

trainTargets, trainImages, testTargets, testImages = data
trainTargets[trainTargets == 3] = 1
trainTargets[trainTargets == 7] = -1

# print(np.unique(trainTargets)); assert 0
trainImages = np.sign(trainImages)
trainImages[trainImages == 0] = -1


print(np.unique(trainImages), np.unique(trainTargets))
idx = np.random.permutation(len(trainTargets))
trainTargets = trainTargets[idx, :]
trainImages  = trainImages[idx, :]
model = mlp(trainImages, trainTargets, M= 100, eta = 1e-5)
# fw = model.updateForward(model.X)
# model.updateBack(model.targets, fw)
tmp = model.train(800)
print(tmp[-1])
# errors, pred = model.train(num = 10)
#
fig, ax  = subplots()
ax.plot(tmp)
show()

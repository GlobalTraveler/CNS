# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 11:17:14 2016

@author: Casper van Elteren
"""

from pylab import *
import numpy as np


# produces output
def act_func(x):
    return x
def perceptron(patterns, targets, weights, moment, eta, gamma, threshold,\
            maxiter, activation_function = act_func):
    # prevent aliasing
    weights = weights.copy()
    if moment:
        momentum = zeros(weights.shape);

    # set initial parameters
    errors = [1];
    # keep track of iterations needed
    idx = 0;
    while abs(errors[-1]) > threshold:
        # weighted sum of input
        h = patterns.dot(weights);

        # compute activation
        tHat = activation_function(h);
        # compute the error
        error = targets - tHat;
        # update rules according to eq 58 (p.76)
        if moment:
            delta_weights =  eta * patterns.T.dot(error);
            weights += gamma * momentum  + delta_weights;
            momentum = delta_weights;
        else:
            weights += eta * patterns.T.dot(error)
        # calculate the squared error
        errors.append(.5*np.sum(error**2))
        idx += 1;
        # have a break point if not lin separable
        if  idx >= maxiter:
            break
    return errors, idx

def stochastic_perceptron(patterns, targets, weights, moment, eta, gamma, threshold,\
            maxiter, activation_function = act_func):
    n_patterns = patterns.shape[0]
    index = list(range(n_patterns))
    # prevent aliasing
    weights = weights.copy()
    # only use this when moment is activated
    if moment:
        momentum = zeros(weights.shape);
    error = 1;
    idx = 0
    errors = []
    while abs(error) > threshold:
        # shuffle the indices
        np.random.shuffle(index);
        # obtain both shuffled data and targets
        shuffleData = patterns[index,:];
        shuffleTargets = targets[index]
        sumerrors = 0;
        # run through all the patterns one by one and update weights
        for trial in range(n_patterns):
            # pick a random index
            # get the sample
            sample = shuffleData[trial,:];
            # get the corresponding target
            target = shuffleTargets[trial];
            # compute the activation
            activation = np.array(activation_function(sample.dot(weights)), ndmin = 2)
            # compute the error
            error = target - activation;
            # update the weights
            if moment:
                delta_weights = eta *(error * sample).T;
                weights += delta_weights + gamma * momentum;
                momentum = delta_weights;
            else:
                weights += eta * (error * sample).T
            sumerrors += error**2
        errors.append(.5*sumerrors)
        error = sumerrors
        idx += 1;
        if idx > maxiter:
             break
    return errors, idx



def genData(n_patterns, n_neurons, n_out):
    # this generates random data for n_neurons, the +1 here is the bias term added
    # create random patterns
#==============================================================================
#     patterns = np.random.rand(n_patterns, n_neurons + 1) * 2 - 1;
#     targets = np.random.rand(n_patterns, n_out)*2 - 1;
#==============================================================================
    # initialize random weights
    weights = np.random.rand(n_neurons + 1, n_out)* 2 - 1;

    patterns = np.random.randint(0,2,(n_patterns, n_neurons + 1))
    targets = np.random.randint(0,2,(n_patterns, n_out))
    patterns [patterns == 0] = -1;
    targets [targets == 0] = -1;


    return patterns, targets, weights


if __name__ == '__main__':
    # note to self: if the eta is too low in combination with the maxiter, the algorithm will not converge. Conversely, if i set eta to be >= 1, the algorithm will blow up
    n_neurons = 50;
    n_out = 1;
    # use momentum
    moment = 0;
    maxiter = 100;
    eta = .01;
    gamma = .1;
    threshold = 1e-2;
    # gen a pattern range
    dstep = 2;
    patternRange = [x for x in range(1, n_neurons*2, dstep)];
    # error vector
    # number of algorithms used
    nAlg = 4;
    errors = np.zeros((nAlg,len(patternRange)));
    iterations = np.zeros((nAlg,len(patternRange)));

    for idx, patterni in enumerate(patternRange):
        if not idx % 10:
            print('Busy on pattern %d' %(patterni))
        # generate random data
        patterns, targets, weights = genData(patterni, n_neurons, n_out);

        # no momentum
        res =  perceptron(patterns, targets, weights, 0, eta, gamma, threshold, maxiter);
        errors[0, idx]  = res[0][-1]; iterations[0, idx] = res[1];

        # momentum
        res =  perceptron(patterns, targets, weights, 1, eta, gamma, threshold, maxiter);
        errors[1, idx] = res[0][-1]; iterations[1, idx] = res[1];

        # stochastic performs slightly bettter than no momentum
        res = stochastic_perceptron(patterns,targets, weights, 0, eta, gamma, threshold, maxiter)
        errors[2, idx] = res[0][-1]; iterations[2, idx] = res[1];

        # stochastic learning with moment
        res = stochastic_perceptron(patterns,targets, weights, 1, eta, gamma, threshold, maxiter)
        errors[3, idx] = res[0][-1]; iterations[3, idx] = res[1];

    patternRange = np.array(patternRange) / n_neurons
    #%%
    # plot the sum squared error as a function of the number of patterns
    figure(1); clf()
    # plot the sum of squared errors
    subplot(211)
    semilogy(patternRange, errors.T)
    plot(np.ones(patternRange[-1])* threshold,'--r')

    # formatting
    legend(['Batch','Batch + Moment','Stochastic','Stochastic + moment','Threshold'], loc ='best', ncol = 2)
    ylabel('Sum squared error'); xlabel('Num patterns / num neurons');
    title('squared error as a function of patterns \n' + \
     'eta = %.1e, gamma = %.1e, n_neurons = %d, maxiter = %d' \
     %(eta, gamma, n_neurons, maxiter))

    # plot the number of iterations needed
    subplot(212)
    plot(patternRange, iterations.T)

    # formatting
    legend(['-Moment','+Moment','Stochastic','Moment stochastic'], loc = 'best')
    xlabel('Num patterns / num neurons'); ylabel('Number of iterations')
    savefig('../Figures/fig1')

    show()

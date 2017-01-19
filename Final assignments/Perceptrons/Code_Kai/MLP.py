import numpy as np
import sys
import pickle
import copy

class MLP():

	act_functions = {'sig': (lambda x:  1 / (1 + np.exp(-x))),\
	 'tanh': (lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))),\
	  'relu': (lambda x: np.maximum(x,0,x))}

	derivatives = {'sig': (lambda x: x * (1-x)), 'tanh': (lambda x: 1 - x**2), 'relu': (lambda x: 1 * (x > 0))}

	'''
	Constructor
	@param layers: List containing the number of neurons per layer
	@param eta: learning rate
	@param printname: file name for parameter saving
	@param alpha: momentum weight
	@param act_fun: String to choose an activation function
	'''
	def __init__(self, layers, eta, printname='nuex', alpha = 0, act_fun='sig'):

		#lrange = lambda s1,s2: np.sqrt(6/(s1+1+s2))
		#self.weights = [np.random.uniform(-lrange(layers[i],layers[i+1]),lrange(layers[i],layers[i+1]),size=(layers[i]+1, layers[i+1])) for i in range(0, len(layers)-1)]
		
		self.weights = [np.random.randn(layers[i]+1, layers[i+1]) for i in range(0, len(layers)-1)]
		self.act_fun = MLP.act_functions[act_fun] 
		self.derivative = MLP.derivatives[act_fun]
		self.eta = eta
		self.printname = printname
		self.error_over_time = []
		self.clsfe_over_time = []
		self.Wabs = []
		self.alpha = alpha
		self.old_deltas = [np.zeros((n.shape[1],n.shape[0]-1)) for n in self.weights]

	'''
	Method to set network weights.
	@param weights: List of weight matrices
	'''
	def set_weights(self, weights):

		self.weights = weights

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def dsoftmax(self, x, dy):
			dx = x * dy
			s = dx.sum(axis=dx.ndim - 1, keepdims=True)
			dx -= x * s

			return dx

	'''
	Method to calculate forward activations.
	Must be called before backwards call for correct calculations.
	@param data: sample
	@return List of outputs per layer
	'''
	def forward(self, data):
		self.outputs = [data]
		for l, layer in enumerate(self.weights):
			#add bias
			out = np.append(np.ones((self.outputs[l].shape[0],1)), self.outputs[l], axis=1)
			self.outputs.append(self.act_fun(np.dot(out,layer)))
		
		return self.outputs


	'''
	Method to back-propagate error through the network and adjust weights.
	Must be called after forward call for correct calculations.
	@param targets
	'''
	def backward(self, targets):

		# Calculate deltas
		error = (targets - self.outputs[-1])
		deltas = [self.derivative(self.outputs[-1]) * error]
		for k in range(len(self.weights) - 1, 0, -1):
			derivative = self.derivative(self.outputs[k])
			partial_error = np.dot(deltas[0],self.weights[k].T)
			deltas.insert(0,partial_error[:,1:] * derivative)
			

		# Adjust weights
		for l, layer in enumerate(self.weights):
			self.old_deltas[l] = (np.dot(deltas[l].T,self.outputs[l]) * self.eta) + self.alpha*self.old_deltas[l]
			layer[1:] += self.old_deltas[l].T
			layer[0] += np.mean(deltas[l],axis=0) * self.eta



	'''
	Method to calculate sum of weights
	@return sum of absolute weights
	'''
	def sumWeights(self):
		return np.sum([np.sum(np.abs(W)) for W in self.weights])

	'''
	Calculate maximum absolut weight change
	@return change
	'''
	def weightChange(self, old_weights):
		return np.max([np.max(np.abs(w1-w2)) for w1,w2 in zip(self.weights, old_weights)])
	'''
	Training cylce
	@param data: Collection of data samples
	@param targets: Collection of target values
	@param epochs: Number of iterations over all data points
	@param samples: Number of samples per epoch
	@param stopTol: Tolerance for weight changes to stop training
	@param save: Save training paramters after training
	@return error per epoch; accuracy per epoch
	'''
	def train(self, data, targets, epochs, samples, batch_size=10, stopTol=1e-5, save=False):
				
		print('Training {} epochs with {} samples each.'.format(epochs, samples))

		for epoch in range(epochs):
			sys.stdout.write('Epoch ' + str(epoch+1) + '\n')
			oldW = copy.deepcopy(self.weights)
			e = []
			c = []
			for index in np.random.choice(np.arange(len(data)), size=(samples//batch_size, batch_size), replace=False):
				self.forward(np.vstack(data[index]))
				e.append(np.mean((self.outputs[-1] - targets[index])**2))
				c.append(np.sum(targets[index][0]==np.round(self.outputs[-1][0])))
				self.backward(np.vstack(targets[index]))
			
			self.error_over_time.append(np.mean(e,axis=0))
			self.clsfe_over_time.append(np.mean(c,axis=0))
			#self.Wabs.append(self.sumWeights())
			print('accuracy: {}'.format(self.clsfe_over_time[-1]))
			print('error: {}'.format(self.error_over_time[-1]))

			dW = self.weightChange(oldW)
			print(dW)
			if dW <= stopTol: break
		
		print('Finished.')
		
		if save:
			filename = 'weights_eta_' + self.printname + '.pkl'
			pickle.dump(self.weights,open(filename,'wb'))
			filename = 'error_eta_' + self.printname + '.pkl'
			pickle.dump(self.error_over_time,open(filename,'wb'))
			filename = 'clsfe_eta_' + self.printname + '.pkl'
			pickle.dump(self.clsfe_over_time,open(filename,'wb'))
			filename = 'wChanges_eta_' + self.printname + '.pkl'
			pickle.dump(self.Wabs,open(filename,'wb'))

		return self.error_over_time, self.clsfe_over_time

	'''
	Method to test network performance
	@param data: test samples
	@param targets: target values
	@return squared error per sample, accuracy
	'''
	def test(self, data, targets):
		
		error = []
		c = []
		for i, d in enumerate(data):
			error.append((self.forward(d[None,:])[-1] - targets[i])**2)
			c.append(targets[i][0]==round(self.outputs[-1][0,0]))
		return error, np.sum(c)/len(data)


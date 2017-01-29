import numpy as np
import scipy
import sys
import pickle
import copy

class MLP():

	#Activation functions
	act_functions = {'sig': (lambda x:  1 / (1 + np.exp(-x))),\
	 'tanh': (lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))),\
	 'tanh_opt': (lambda x: 1.7159 * np.tanh(2/3 * x)),\
	 'relu': (lambda x: np.maximum(x,0,x))}

	#Derivatives
	derivatives = {'sig': (lambda x: x * (1-x)), 'tanh': (lambda x: 1 - x**2),\
	 'tanh_opt': lambda x: 2/3*1.7159*(1 - 1/1.7159**2*x**2),'relu': (lambda x: 1 * (x > 0))}

	'''
	Constructor
	@param layers: List containing the number of neurons per layer
	@param eta: learning rate
	@param printname: file name for parameter saving
	@param alpha: momentum weight
	@param act_fun: String to choose an activation function
	'''
	def __init__(self, layers, eta, printname='nuex', alpha = 0, act_fun='sig'):

		self.weights = [np.random.randn(layers[i]+1, layers[i+1]) for i in range(0, len(layers)-1)]
		self.act_name = act_fun
		self.act_fun = MLP.act_functions[act_fun] 
		self.derivative = MLP.derivatives[act_fun]
		self.eta = eta
		self.printname = printname
		self.error_over_time = []
		self.clsfe_over_time = []
		self.Wabs = []
		self.alpha = alpha
		

		if 'tanh' in act_fun:
			self.comp_out = lambda t,y: t==np.sign(y)
			self.final_layer = lambda x: self.act_fun(x)
			self.final_derivative = lambda x: self.derivative(x)
		else:
			self.comp_out = lambda t,y: t==np.round(y)
			self.final_layer = lambda x: MLP.act_functions['sig'](x)	
			self.final_derivative = lambda x: MLP.derivatives['sig'](x)
		

		self.old_deltas = [np.zeros((n.shape[1],n.shape[0])) for n in self.weights]
		self.g_old = np.vstack([np.ones((n.shape[1]*(n.shape[0]),1)) for n in self.weights])
		self.g0 = np.zeros(self.g_old.shape)
	'''
	Method to set network weights.
	@param weights: List of weight matrices
	'''
	def set_weights(self, weights):
		self.weights = weights

	'''
	Method to calculate forward activations.
	Must be called before backwards call for correct calculations.
	@param data: sample
	@return List of outputs per layer
	'''
	def forward(self, data):
		
		self.outputs = [data]
		for l, layer in enumerate(self.weights[:-1]):
			#add bias
			self.outputs[l] = np.append(np.ones((self.outputs[l].shape[0],1)), self.outputs[l], axis=1)
			out = self.act_fun(np.dot(self.outputs[l],layer))
			
			self.outputs.append(out)
		
		#Output layer activity
		l += 1
		layer = self.weights[-1]
		self.outputs[l] = np.append(np.ones((self.outputs[l].shape[0],1)), self.outputs[l], axis=1)
		self.outputs.append(self.final_layer(np.dot(self.outputs[l],layer)))
		
		return self.outputs

	'''
	Method to back-propagate error through the network and adjust weights.
	Must be called after forward call for correct calculations.
	@param targets
	'''
	def backward(self, targets, method):

		# Calculate deltas
		deltas = self.calcDeltas(targets)

		#Stochastic gradient descent
		if method=='sgd':	
			
			# Adjust weights
			batch_size = self.outputs[0].shape[0]
			for l, layer in enumerate(self.weights):
				self.old_deltas[l] = (1/batch_size * np.dot(deltas[l].T,self.outputs[l]) * self.eta) + self.alpha*self.old_deltas[l]
				layer += self.old_deltas[l].T

		#Conjugate gradient descent
		else:

			if np.linalg.norm(self.g_old) <= self.epsilon*np.linalg.norm(self.g0):
				return

			#Find update step size with line search
			opt = scipy.optimize.minimize_scalar(fun=lambda x: self.cost(x,targets), method='Brent').x
			# Adjust weights
			for l, layer in enumerate(self.weights):
				layer +=  opt * self.old_deltas[l].T
			
			#Find gradient direction
			gs = []
			ds = []
			for l, layer in enumerate(self.weights):
				g = np.dot(deltas[l].T,self.outputs[l])
				ds.append(-g)
				gs.append(g.flatten()[:,None])

			gs = np.vstack(gs)
			
			#Find beta weight for conjugacy 
			norm = np.dot(self.g_old.T,self.g_old) + 1e-14
			beta = np.dot(gs.T,(gs-self.g_old)) / norm
			beta = np.maximum(beta,0)


			#Update direction
			self.g_old = gs
			for l, layer in enumerate(self.weights):
				new_delta = ds[l] + beta*self.old_deltas[l]
				self.old_deltas[l] = new_delta

			if np.all(self.g0==0):
				self.g0 = self.g_old

	'''
	Method to calculate partial derivatives
	@param targets
	@return partial derivatives
	'''
	def calcDeltas(self, targets):
		# Calculate deltas
		error = (targets - self.outputs[-1])
		deltas = [self.final_derivative(self.outputs[-1]) * error]
		i = 0
		for k in range(len(self.weights) - 1, 0, -1):
			derivative = self.derivative(self.outputs[k][:,1:])
			partial_error = np.dot(deltas[0],self.weights[k][1:,:].T)
			delta = partial_error * derivative
			deltas.insert(0,delta)

		return deltas	


	'''
	Cost Function for line search for conjugate gradient descent
	@param b: line length
	@param targets
	@return: mean squared error cost
	'''
	def cost(self, b, targets):
		outputs = [self.outputs[0][:,1:]]
		for l, layer in enumerate(self.weights):
			#add bias
			out = np.append(np.ones((outputs[l].shape[0],1)), outputs[l], axis=1)
			outputs.append(self.act_fun(np.dot(out,layer+b*self.old_deltas[l].T)))

		return np.mean((outputs[-1]-targets)**2) 


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
		return np.mean([np.mean(np.abs(w1-w2)) for w1,w2 in zip(self.weights, old_weights)])
	
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
	def train(self, data, targets, epochs, samples, batch_size=10, stopTol=1e-5, val=0, method='sgd', save=False):
		
		self.epsilon = stopTol				
		#If validation set is to be use, split data
		if val:
			mask = np.random.choice(np.arange(len(data)), size=int(val * samples), replace=False)
			valData = data[mask]
			valTargets = targets[mask]

			data = np.delete(data,mask,0)
			tagets = np.delete(targets,mask,0)

			if batch_size==samples: batch_size = int((1-val) * batch_size)
			
			samples = int((1-val)*samples)	

		print('Training {} epochs with {} samples each.'.format(epochs, samples))
		count = 0
		countTest = 0 
		e0 = self.eta

		testErrors = []
		dE = np.inf
		for epoch in range(epochs):
			sys.stdout.write('Epoch ' + str(epoch+1) + '\n')
			oldW = copy.deepcopy(self.weights)
			e = []
			c = []
			for index in np.random.choice(np.arange(len(data)), size=(samples//batch_size, batch_size), replace=False):
				self.forward(np.vstack(data[index]))
				
				e.append(np.mean((self.outputs[-1] - targets[index])**2))
				c.append(np.sum(self.comp_out(targets[index],self.outputs[-1])))
				
				self.backward(np.vstack(targets[index]), method)

			#self.eta = e0*np.exp(-e0*epoch)

			self.error_over_time.append(np.mean(e,axis=0))
			self.clsfe_over_time.append(np.sum(c,axis=0) / samples)
			self.Wabs.append(self.sumWeights())
			print('accuracy: {}'.format(self.clsfe_over_time[-1]))
			print('error: {}'.format(self.error_over_time[-1]))

			if val:
				t,_ = self.test(valData,valTargets)
				t = np.mean(t)
				dE = testErrors[-1] - t
				testErrors.append(t)

			dW = self.weightChange(oldW)
			print('delta W: {}'.format(dW))
			#Convergence if weight change smaller than stopTol for 3 epochs
			if epoch > 1 and dW <= stopTol: 
				count += 1
			else:
				count = 0
			if count >= 5: break


			#Convergence if validation does not go down anymore
			if epoch > 1 and dE <= 1e-3: 
				countTest += 1
			else:
				countTest = 0
			if countTest >= 20: break
		
		print('Finished.')
		
		if save:
			filename = 'weights_' + self.printname + '.pkl'
			pickle.dump(self.weights,open(filename,'wb'))
			filename = 'error_' + self.printname + '.pkl'
			pickle.dump(self.error_over_time,open(filename,'wb'))
			filename = 'valError_' + self.printname + '.pkl'
			pickle.dump(self.error_over_time,open(filename,'wb'))
			filename = 'clsfe_' + self.printname + '.pkl'
			pickle.dump(self.clsfe_over_time,open(filename,'wb'))
			filename = 'wChanges_' + self.printname + '.pkl'
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
			c.append(self.comp_out(targets[i],self.outputs[-1]))
		return error, np.sum(c)/len(data)


import numpy as np
import sys
import pickle
import argparse
from MLP import MLP
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat

def showData(x):
	imdim = np.sqrt(x.shape[0])
	plt.imshow(np.reshape(x,(imdim,imdim)).T)
	plt.show()


def plot_error(error, accuracy, args, layers):
	
	if not error:
		error = pickle.load(open('error_eta_{}.pkl'.format(args.f_name),'rb'))
	if not accuracy:
		accuracy = pickle.load(open('clsfe_eta_{}.pkl'.format(args.f_name),'rb'))

	m = np.array(error)
	a = np.array(accuracy) 
	x = np.arange(0,len(error),1)

	fig, ax = plt.subplots(1,2,sharex=True,figsize=(15,7))
	fig.suptitle('Learning on training data\n Differentiate 3s and 7s\n $\eta = {}, momentum = {}, neurons = {}$'.format(args.eta, args.alpha, layers))
	ax[0].plot(x,m)
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel('Mean squared error')
	ax[1].plot(x,a)
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Accuracy')
	fig.savefig('{}_eta_{}_alpha_{}_layers{}.pdf'.format(args.f_name,args.eta,args.alpha,layers))

def loadData(nums = (3,7), oneHot=False):


	mnist = loadmat('../Data/minst.mat')
	xtrain = mnist['mnist'][0][0][0] 
	ytrain = mnist['mnist'][0][0][2]

	xtest = mnist['mnist'][0][0][1]
	ytest = mnist['mnist'][0][0][3]

	xtrain_all = []
	xtest_all = []
	ytrain_all = []
	ytest_all = []
	for i,num in enumerate(nums):

		#train images
		x = xtrain[:,:,ytrain[:,0]==[num]]


		n = x.shape[0]**2 
		#flatten image dimensions
		x = np.reshape(x,(n,x.shape[2]))
		x = np.double(x)

		xtrain_all.append(x.T)
		ytrain_all.append(np.ones((x.shape[1],1))*i)

		#test images
		x = xtest[:,:,ytest[:,0]==[num]]

		n = x.shape[0]**2 
		#flatten image dimensions
		x = np.reshape(x,(n,x.shape[2]))
		x = np.double(x)

		xtest_all.append(x.T)
		ytest_all.append(np.ones((x.shape[1],1))*i)

	if oneHot:
		enc = OneHotEncoder()
		ytrain = enc.fit_transform(ytrain).toarray()
		ytest = enc.fit_transform(ytest).toarray()


	return np.vstack(xtrain_all), np.vstack(ytrain_all), np.vstack(xtest_all), np.vstack(ytest_all)

def main(args):


	input_dim = args.input_dim
	output_dim = args.output_dim
	layers = args.hidden
	layers.insert(0,input_dim)
	layers.append(output_dim)
	eta = args.eta
	epochs = args.epochs
	batch_size = args.batch_size
	sample_size = args.sample_size * 2
	alpha = args.alpha 
	stopTol = args.stopTol
	act_function = args.act_function
	f_name = args.f_name

	train_data, train_targets, test_data, test_targets = loadData()

	#Normalize
	if act_function == 'tanh':
		#treshold value for binarization
		threshold = 0
		train_data = 2*(train_data > threshold) - 1
		test_data = 2*(test_data > threshold) - 1
	else:
		train_data /= 255
		test_data /= 255
	# showData(new_data[15])

	multi = MLP(layers,eta,f_name,alpha, act_fun=act_function)

	# Train network
	save = True if args.plot else False
	error, accuracy = multi.train(train_data, train_targets, epochs, sample_size, batch_size=batch_size, stopTol=stopTol, save=save)

	# Plot learning curves
	if args.plot: 
		plot_error(error, accuracy, args, layers)

	# Test network
	error,accuracy = multi.test(test_data, test_targets)

	m = np.mean(np.array(error))
	s = np.sum(np.mean(np.array(error),axis=0))
	print('Performance on Testdata:')
	print('Mean squared error is ', m)
	print('Sum of squares error is ', s)
	print('Percentage of correctly classified imgs is ',accuracy)

	return accuracy

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a multi layer perceptron to classify MNIST digits')
	parser.add_argument('--input_dim', action='store', type=int, default=784,
                    help='Input dimension')
	parser.add_argument('hidden', nargs='+', type=int,
                    help='Number of hidden units')
	parser.add_argument('--output_dim', action='store', type=int, default=1,
                    help='Output dimension')
	parser.add_argument('-r', action='store', dest='eta', type=float, default=0.0001,
                    help='MLP learning rate')
	parser.add_argument('-e', action='store', dest='epochs', type=int, default=20,
                    help='Number of training epochs')
	parser.add_argument('-b', action='store', dest='batch_size', type=int, default=1,
                    help='Size of mini batches')
	parser.add_argument('-m', action='store', dest='alpha', type=float, default=0,
                    help='Alpha for momentum')
	parser.add_argument('-n', action='store', dest='sample_size', type=int, default=6000,
                    help='Number of training samples per class; max=6000')
	parser.add_argument('-t', action='store', dest='stopTol', type=float, default=1e-5,
                    help='Tolerance of weight changes for convergence')
	parser.add_argument('-f', action='store', dest='f_name',
                    help='File ending for weight saving')
	parser.add_argument('-a', action='store', dest='act_function', default='sig', choices=('sig', 'tanh', 'relu'),
                    help='Activation function')	
	parser.add_argument('--plot', action='store_true', dest='plot', default=False,
                    help='Plot results')	
	args = parser.parse_args()

	main(args)
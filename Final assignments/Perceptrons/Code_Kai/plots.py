import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []

args = Namespace(input_dim=784, output_dim=1, hidden=[20], eta=0.01,\
 	epochs=1000, sample_size=6000, batch_size=32, alpha=0, \
 	act_function='sig', stopTol=1e-5, conj=False, validation=0, f_name='SGD', plot=False)

results_sgd = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[20], eta=0.01,\
 	epochs=1000, sample_size=6000, batch_size=32, alpha=0.8, \
 	act_function='sig', stopTol=1e-5, conj=False, validation=0, f_name='SGD_mom', plot=False)

results_sgd_mom = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[20], eta=0.1,\
 	epochs=1000, sample_size=6000, batch_size=12000, alpha=0.8, \
 	act_function='sig', stopTol=1e-5, conj=False, validation=0, f_name='batch', plot=False)

results_batch = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[20], eta=0.01,\
 	epochs=1000, sample_size=6000, batch_size=32, alpha=0, \
 	act_function='sig', stopTol=1e-5, conj=True, validation=0, f_name='conj', plot=False)

results_conj = train_set.main(args)

print('Test Accuracies')
print('SGD: {}, SGD+Momentum: {}, Batch: {}, Conj: {}'\
	.format(results_sgd[3], results_sgd_mom[3], results_batch[3], results_conj[3]))

print('Train Accuracies')
print('SGD: {}, SGD+Momentum: {}, Batch: {}, Conj: {}'\
	.format(results_sgd[1][-1], results_sgd_mom[1][-1], results_batch[1][-1], results_conj[1][-1]))

print('Train Error')
print('SGD: {}, SGD+Momentum: {}, Batch: {}, Conj: {}'\
	.format(results_sgd[0][-1], results_sgd_mom[0][-1], results_batch[0][-1], results_conj[0][-1]))

print('#Iterations')
print('SGD: {}, SGD+Momentum: {}, Batch: {}, Conj: {}'\
	.format(len(results_sgd[0]), len(results_sgd_mom[0]), len(results_batch[0]), len(results_conj[0])))

f = plt.figure()
plt.plot(results_sgd[0], label='stochastic grad')
plt.plot(results_sgd_mom[0], label='stochastic grad + mom')
plt.plot(results_batch[0], label = 'batch grad')
plt.plot(results_conj[0], label = 'conj grad')

plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.title('Cost as function of iterations')
plt.legend()
f.savefig('comp.pdf')
plt.show()
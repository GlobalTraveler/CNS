import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []

args = Namespace(input_dim=784, output_dim=1, hidden=[10], eta=0.001,\
 	epochs=100, sample_size=6000, batch_size=32, alpha=0, \
 	act_function='tanh_opt', stopTol=1e-5, dropout=0, conj=False, f_name='SGD', plot=False)

results_sgd = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[10], eta=0.001,\
 	epochs=100, sample_size=6000, batch_size=32, alpha=0.8, \
 	act_function='tanh_opt', stopTol=1e-5, dropout=0, conj=False, f_name='SGD_mom', plot=False)

results_sgd_mom = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[10], eta=0.1,\
 	epochs=100, sample_size=6000, batch_size=12000, alpha=0.7, \
 	act_function='tanh_opt', stopTol=1e-5, dropout=0, conj=False, f_name='batch', plot=False)

results_batch = train_set.main(args)

args = Namespace(input_dim=784, output_dim=1, hidden=[10], eta=0.01,\
 	epochs=100, sample_size=6000, batch_size=32, alpha=0, \
 	act_function='tanh_opt', stopTol=1e-10, dropout=0, conj=True, f_name='conj', plot=False)

results_conj = train_set.main(args)


plt.plot(results_sgd[0], label='stochastic grad')
plt.plot(results_sgd_mom[0], label='stochastic grad + mom')
plt.plot(results_batch[0], label = 'batch grad')
plt.plot(results_conj[0], label = 'conj grad')

plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.title('Cost as function of iterations')
plt.legend()
plt.show()
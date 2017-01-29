import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []
convergence = []

etas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

for eta in etas:

	args = Namespace(input_dim=784, output_dim=1, hidden=[20], eta=eta,\
 	epochs=1000, sample_size=6000, batch_size=32, alpha=0.8, \
 	act_function='sig', stopTol=1e-5, conj=False, validation=0.05, f_name='SGD_mom', plot=False)

	e,_,_,accuracy = train_set.main(args)
	
	accuracies.append(accuracy)
	convergence.append(len(e))

pickle.dump(accuracies, open('etaAccuracies.pkl','wb'))
pickle.dump(convergence, open('etaConvergence.pkl','wb'))

plt.plot(np.arange(len(etas)), accuracies)
plt.xlabel('$\eta$')
plt.xticks(np.arange(len(etas)),etas)
plt.ylabel('Test accuracy')
plt.title('Test accuracy dependent on learning rate')
plt.show()
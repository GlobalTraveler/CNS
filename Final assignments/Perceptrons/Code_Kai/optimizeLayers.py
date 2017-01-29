import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []
astd = []
convergence = []
cstd = []
n = list(range(1,6))
for layer in n:
	
	acs = []
	convs = []
	np.random.seed(1234)

	for i in range(10):
		args = Namespace(input_dim=784, output_dim=1, hidden=[50] * layer, eta=0.1,\
	 	epochs=1000, sample_size=6000, batch_size=128, alpha=0.8, \
	 	act_function='sig', stopTol=1e-3, conj=False, validation=0, f_name='SGD_mom', plot=False)
		

		e,_,_,accuracy = train_set.main(args)
		acs.append(accuracy)
		convs.append(len(e))
	
	accuracies.append(np.mean(acs))
	astd.append(np.std(acs))
	convergence.append(np.mean(convs))
	cstd.append(np.std(convs))

pickle.dump(accuracies, open('layerAccuracies.pkl','wb'))
pickle.dump(convergence, open('layerConvergence.pkl','wb'))
pickle.dump(astd, open('layerAccuraciesSTD.pkl','wb'))
pickle.dump(cstd, open('layerConvergenceSTD.pkl','wb'))


f, ax = plt.subplots(1,2, sharex=True)

ax[0].plot(n, accuracies)
ax[1].plot(n, convergence)
ax[0].set_xlabel('#Layers')
ax[1].set_xlabel('#Layers')
ax[0].set_ylabel('Test accuracy')
ax[1].set_ylabel('Iteration until convergence')
f.suptitle('Test accuracy dependent on number of hidden layers')
f.savefig('optLayers.pdf')
plt.show()
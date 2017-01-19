import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []
etas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
for eta in etas:

	args = Namespace(input_dim=784, output_dim=1, hidden=[100], eta=eta, epochs=10, sample_size=6000, alpha=0, f_name='test', plot=False)

	accuracy = train_set.main(args)
	accuracies.append(accuracy)

pickle.dump(accuracies, open('etaAccuracies.pkl','wb'))
plt.plot(np.arange(len(etas)), accuracies)
plt.xlabel('$\eta$')
plt.xticks(np.arange(len(etas)),etas)
plt.ylabel('Test accuracy')
plt.title('Test accuracy dependent on learning rate')
plt.show()
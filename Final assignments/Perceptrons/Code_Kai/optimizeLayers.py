import numpy as np
from argparse import Namespace
import train_set
import matplotlib
import matplotlib.pyplot as plt
import pickle

accuracies = []
r = range(1,7)
for nLayers in r:

	args = Namespace(input_dim=784, output_dim=1, hidden=[100//nLayers]*nLayers, eta=0.001, epochs=10, sample_size=6000, alpha=0, f_name='test', plot=False)

	accuracy = train_set.main(args)
	accuracies.append(accuracy)

pickle.dump(accuracies, open('layersAccuracies.pkl','wb'))
plt.plot(r, accuracies)
plt.xlabel('#Layers')
#plt.xticks(np.arange(len(etas)),etas)
plt.ylabel('Test accuracy')
plt.title('Test accuracy dependent on number of layers')
plt.show()
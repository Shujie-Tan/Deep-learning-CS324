import numpy as np
import matplotlib.plot as plt

def Gauss2d(mean, cov, c):
    x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
    x1 = x1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    label = np.array([c] * 100)
    label = label[:, np.newaxis]
    return np.concatenate((x1,x2,label), axis=1)

X = []
mean1 = (1, 2)
cov1 = [[1, 0], [0, 1]]
x1 = Gauss2d(mean1, cov1, -1)
mean2 = (3, 4)
cov2 = [[1,0], [0, 1]]
x2 = Gauss2d(mean2, cov2, 1)
X = np.concatenate((x1,x2), axis=0)

np.random.shuffle(X)
training, test = X[:160,:], X[160:,:]

label = training[:, -1]
plt.plot(training[label==-1.][:,0], training[label==-1.][:,1], 'x')
plt.plot(training[label==1.][:,0], training[label==1.][:,1], 'o')
plt.axis('equal')
plt.show()



class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.W1 = np.zeros(())
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """

        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
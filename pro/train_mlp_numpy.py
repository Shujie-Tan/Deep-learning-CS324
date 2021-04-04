from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp import MLP
from modules import CrossEntropy
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '3'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    correct = 0
    for i in range(len(labels)):
        if (predictions[i] == labels[i]).all( ):
            correct += 1
    accuracy = correct / len(labels)
    return accuracy


def train(model, X_train, y_train, X_test, y_test):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    acc = []
    cross_entropy = CrossEntropy()
    for epoch in range(FLAGS.max_steps):
        np.random.shuffle(X_train)
        for i in range(len(X_train)):
            out = model.forward(X_train[i])
            loss = cross_entropy.forward(out, y_train[i])
            dout = cross_entropy.backward(out, y_train[i])
            model.backward(dout)
            # gradient descent
            for l in range(len(model.linear)):
                model.linear[l].weight -= FLAGS.learning_rate * model.linear[l].grads['weight']
            #    print(FLAGS.learning_rate)
             #   print(model.linear[l].grads['weight'])
              #  print(model.linear[l].weight)
                model.linear[l].bias -= FLAGS.learning_rate * model.linear[l].grads['bias']

        if epoch % FLAGS.eval_freq == 0:

            predictions = np.zeros(y_test.shape)
            for xi, x in enumerate(X_test):
                predictions[xi][np.argmax(model.forward(x))] = 1
            acc.append(accuracy(predictions, y_test))
            print("the accuracy of epoch {} is {}".format(epoch, acc[epoch // FLAGS.eval_freq]))


def main():
    """
    Main function
    """
    X, y = datasets.make_moons(1000, noise=0.10)
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.asarray(list(set(y))).reshape(-1, 1))
    y = enc.transform(np.asarray(y).reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    n_inputs = X.shape[1]
    n_hidden = [int(s) for s in FLAGS.dnn_hidden_units.split(sep=",")]
    n_classes = 2
    model = MLP(n_inputs, n_hidden, n_classes)
    train(model, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()

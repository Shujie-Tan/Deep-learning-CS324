from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from modules import * 


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        lst = [n_inputs] + n_hidden + [n_classes]
        self.linear = [Linear(lst[i], lst[i+1]) for i in range(len(lst)-1)]
        self.relu = [ReLU() for _ in range(len(n_hidden))]
        self.softmax = SoftMax()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for i in range(len(self.relu)):
            out = self.relu[i].forward(self.linear[i].forward(out))
        out = self.softmax.forward(self.linear[-1].forward(out))
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dout = self.softmax.backward(dout)
        dout = self.linear[-1].backward(dout)
        for i in reversed(range(len(self.relu))):
            dout = self.linear[i].backward(self.relu[i].backward(dout))
        return

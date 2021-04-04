import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution 
        with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.weight = np.random.randn(out_features, in_features) * 0.0001
        self.bias = np.zeros(out_features)
        self.gradients = np.zeros((out_features, in_features))
        self.grads = {}
        self.data = np.array([])
        
    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside 
        the object and use them in the backward pass computation. This is true 
        for *all* forward methods of *all* modules in this class
        """
        self.data = x
        out = np.matmul(self.weight, x) + self.bias
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads['bias'] = np.eye(len(self.bias))

        factor = np.eye(self.weight.shape[0])
        factor = factor[:, np.newaxis]
        factor = np.repeat(factor, len(self.data), axis=1)
        factor = np.transpose(factor, (0, 2, 1))  # kij
        self.grads['weight'] = factor * np.transpose(self.data)
        self.grads['weight'] = np.transpose(self.grads['weight'], (1, 2, 0))  # ijk

        dx = np.matmul(self.grads['weight'], dout)
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.data = x
        out = np.maximum(x, 0)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = 1. * (self.data > 0)
        dx = np.diagflat(dx)
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        x = x - max(x)
        out = np.exp(x) / sum(np.exp(x))
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        out = self.out
        din = np.diagflat(out) - np.outer(out, out)
        dx = np.matmul(din, dout)
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        label = np.argmax(y)
        out = -np.log(x[label] + 1e-15)
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = np.zeros_like(x)
        dx[np.argmax(y)] = -1. / x[np.argmax(y)]
        return dx

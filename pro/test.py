import numpy as np
grad = np.eye(3)
grad = grad[:,np.newaxis]
grad = np.repeat(grad,2,axis=1)
grad = np.transpose(grad,(0,2,1))       # kij
data = np.array([[3],[2]])
grad = grad * np.transpose(data)
grad = np.transpose(grad, (1,2,0))      # ijk   dx/dW

dout = np.array([[2],[3],[4]])  # dout.shape=(3,1)
result = np.matmul(grad,dout)
result = np.squeeze(result)


dout = dout - max(dout)
out = np.exp(dout)  / sum(np.exp(dout))


weight = np.array([[1,2],
                   [3,4],
                   [5,6]])

from modules import *
L = Linear(3,2)
x = np.array([1, 2, 3])
y = L.forward(x)
out = np.array([1,1])
L.backward(out)

dout = np.array([1,0.5,1])
R = ReLU()
x = np.array([-1,1,-3])
R.forward(x)
R.backward(dout)

S = SoftMax()
S.forward(x)
S.backward(dout)
dx = np.diagflat(S.out) - np.matmul(S.out, np.transpose(S.out))
np.matmul(dx, dout)

C = CrossEntropy()
y = np.array([0,1,0])
C.forward(x, y)
C.backward(x,y)

import mlp
from mlp import MLP
import numpy as np
from modules import *
n_inputs = 2
n_hidden = [3, 4]
n_classes = 2
mlp = MLP(n_inputs, n_hidden, n_classes)
x = np.array([1,2])
out = mlp.forward(x)

cross_entropy = CrossEntropy()
y = np.array([0,1])
L = cross_entropy.forward(out, y)
dout = cross_entropy.backward(out, y)
mlp.backward(dout)
mlp.linear[0].grads['weight']

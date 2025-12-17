import numpy as np
from layers.base import Layer


class Dense(Layer):
    def __init__(self, in_features, out_features, initializer):
        self.W = initializer((in_features, out_features))
        self.b = np.zeros(out_features)


    def forward(self, X, training=True):
        self.X = X
        return X @ self.W + self.b


    def backward(self, dY):
        self.dW = self.X.T @ dY
        self.db = dY.sum(axis=0)
        return dY @ self.W.T


    def params(self):
        return [self.W, self.b]


    def grads(self):
        return [self.dW, self.db]
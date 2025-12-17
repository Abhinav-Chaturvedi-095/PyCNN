import numpy as np
from layers.base import Layer


class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate


    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, X.shape)
            return X * self.mask
        return X


    def backward(self, dY):
        return dY * self.mask
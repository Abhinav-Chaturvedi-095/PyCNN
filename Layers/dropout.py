import numpy as np
from layers.base import Layer


class Dropout(Layer):
    def __init__(self, rate=0.5):
        assert 0.0 <= rate < 1.0
        self.rate = rate

    def forward(self, inputs, training=True):
        if not training:
            return inputs

        self.mask = (np.random.rand(*inputs.shape) > self.rate)
        return inputs * self.mask / (1 - self.rate)

    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.rate)

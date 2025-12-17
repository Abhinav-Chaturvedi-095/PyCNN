import numpy as np
from layers.base import Layer


class Dense(Layer):
    def __init__(self, in_features, out_features, initializer, name="Dense"):
        self.W = initializer((in_features, out_features))
        self.b = np.zeros(out_features)

        self.dW = None
        self.db = None
        self.name = name

    def forward(self, inputs, training=True):
        self.inputs = inputs
        return inputs @ self.W + self.b

    def backward(self, grad_output):
        batch_size = grad_output.shape[0]

        self.dW = self.inputs.T @ grad_output / batch_size
        self.db = grad_output.mean(axis=0)

        return grad_output @ self.W.T

    def get_params(self):
        return [
            (self.W, self.dW, f"{self.name}_W"),
            (self.b, self.db, f"{self.name}_b"),
        ]

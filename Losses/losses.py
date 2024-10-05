import numpy as np

class Loss:
    def __init__(self, loss_function, loss_derivative):
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
        self.predicted = None
        self.actual = None

    def forward(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual
        return self.loss_function(self.predicted, self.actual)

    def backward(self):
        return self.loss_derivative(self.predicted, self.actual)

import numpy as np
from Losses.losses import Loss

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        def cce(predicted, actual):
            predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
            return -np.sum(actual * np.log(predicted)) / actual.shape[0]

        def cce_derivative(predicted, actual):
            return (predicted - actual) / actual.shape[0]

        super().__init__(cce, cce_derivative)

import numpy as np
from Losses.losses import Loss

class MeanSquaredError(Loss):
    def __init__(self):
        def mse(predicted, actual):
            return np.mean(np.power(actual - predicted, 2))

        def mse_derivative(predicted, actual):
            return 2 * (predicted - actual) / actual.size

        super().__init__(mse, mse_derivative)

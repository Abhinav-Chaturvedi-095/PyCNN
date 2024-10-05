import numpy as np
from Losses.losses import Loss

class BinaryCrossEntropy(Loss):
    def __init__(self):
        def bce(predicted, actual):
            predicted = np.clip(predicted, 1e-12, 1 - 1e-12)  # Avoid log(0)
            return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

        def bce_derivative(predicted, actual):
            predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
            return -(actual / predicted) + (1 - actual) / (1 - predicted)

        super().__init__(bce, bce_derivative)

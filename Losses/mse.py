import numpy as np
from Losses.losses import Loss

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

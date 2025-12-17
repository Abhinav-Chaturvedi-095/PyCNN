import numpy as np
from Losses.losses import Loss

class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_true = y_true
        return -np.mean(
            y_true * np.log(self.y_pred) +
            (1 - y_true) * np.log(1 - self.y_pred)
        )

    def backward(self):
        return (self.y_pred - self.y_true) / (
            self.y_pred * (1 - self.y_pred)
        )

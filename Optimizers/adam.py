import numpy as np
from Optimizers.optimizers import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, grad, param_name):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(grad)
            self.v[param_name] = np.zeros_like(grad)

        self.t += 1

        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

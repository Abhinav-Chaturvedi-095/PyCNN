from Optimizers.optimizers import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, param, grad, param_name=None):
        param -= self.learning_rate * grad

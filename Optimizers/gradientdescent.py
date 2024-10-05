from Optimizers.optimizers import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, params, gradients):
        for param, grad in zip(params, gradients):
            param -= self.learning_rate * grad

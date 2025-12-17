class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, param, grad, param_name):
        raise NotImplementedError

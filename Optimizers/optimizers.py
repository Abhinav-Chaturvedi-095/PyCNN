class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """
        This method will be overridden by specific optimizers like SGD, Adam, etc.
        It will update the parameters using the gradients.
        
        :param params: The parameters (weights, biases) to be updated.
        :param gradients: The gradients of the loss function with respect to those parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

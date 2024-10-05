import numpy as np
from Activations.activation import Activation

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # For backward pass, derivative is handled with the loss in practice.
        def softmax_derivative(x):
            raise NotImplementedError("Softmax derivative is usually handled with the loss function.")
        
        super().__init__(softmax, softmax_derivative)

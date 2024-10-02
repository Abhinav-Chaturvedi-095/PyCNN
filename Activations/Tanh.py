import numpy as np
from activation import Activation

class Tanh():
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2
        
        super.__init__(tanh,tanh_derivative)
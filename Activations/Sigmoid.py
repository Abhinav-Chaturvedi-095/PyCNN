import numpy as np 
from Activations.activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
    
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        super.__init__(sigmoid,sigmoid_derivative)
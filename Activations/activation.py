import numpy as np

class Activation():
    def __init__(self,activation,activation_derivative) -> None:
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input = None

    def forward(self,input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self,nodeValues):
        return np.multiply(nodeValues,self.activation_derivative(self.input))

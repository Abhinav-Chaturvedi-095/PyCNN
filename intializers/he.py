import numpy as np

def he(shape):
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)

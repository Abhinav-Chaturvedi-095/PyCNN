import numpy as np

def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])
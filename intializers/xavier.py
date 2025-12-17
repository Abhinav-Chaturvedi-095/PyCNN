import numpy as np

def xavier(shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

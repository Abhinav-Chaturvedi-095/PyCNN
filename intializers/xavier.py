import numpy as np

def xavier(shape):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, shape)

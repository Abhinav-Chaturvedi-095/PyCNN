import numpy as np
from layers.base import Layer


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs, training=True):
        self.inputs = inputs
        batch_size, channels, H, W = inputs.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        output = np.zeros((batch_size, channels, H_out, W_out))
        self.max_indices = {}

        for n in range(batch_size):
            for c in range(channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = inputs[
                            n, c, h_start : h_start + k, w_start : w_start + k
                        ]
                        max_val = np.max(region)
                        output[n, c, i, j] = max_val
                        self.max_indices[(n, c, i, j)] = np.argwhere(region == max_val)[0]

        return output

    def backward(self, grad_output):
        dX = np.zeros_like(self.inputs)
        k = self.kernel_size
        s = self.stride

        for (n, c, i, j), idx in self.max_indices.items():
            h_start = i * s
            w_start = j * s
            dX[
                n,
                c,
                h_start + idx[0],
                w_start + idx[1],
            ] += grad_output[n, c, i, j]

        return dX

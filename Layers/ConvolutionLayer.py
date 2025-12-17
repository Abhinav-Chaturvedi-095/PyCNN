import numpy as np
from layers.base import Layer


class Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        initializer=None,
        name="Conv2D",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.name = name

        assert initializer is not None, "Initializer must be provided"

        # Weight shape: (out_channels, in_channels, kH, kW)
        self.W = initializer(
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None

    def forward(self, inputs, training=True):
        self.inputs = inputs
        batch_size, _, H, W = inputs.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        H_out = (H - k + 2 * p) // s + 1
        W_out = (W - k + 2 * p) // s + 1

        # Padding
        X_padded = np.pad(
            inputs,
            ((0, 0), (0, 0), (p, p), (p, p)),
            mode="constant",
        )
        self.X_padded = X_padded

        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = X_padded[
                            n, :, h_start : h_start + k, w_start : w_start + k
                        ]
                        output[n, oc, i, j] = (
                            np.sum(region * self.W[oc]) + self.b[oc]
                        )

        return output

    def backward(self, grad_output):
        batch_size, _, H_out, W_out = grad_output.shape
        _, _, H, W = self.inputs.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        dX_padded = np.zeros_like(self.X_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for n in range(batch_size):
            for oc in range(self.out_channels):
                self.db[oc] += np.sum(grad_output[n, oc])
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = self.X_padded[
                            n, :, h_start : h_start + k, w_start : w_start + k
                        ]
                        self.dW[oc] += grad_output[n, oc, i, j] * region
                        dX_padded[
                            n, :, h_start : h_start + k, w_start : w_start + k
                        ] += grad_output[n, oc, i, j] * self.W[oc]

        if p > 0:
            return dX_padded[:, :, p:-p, p:-p]
        return dX_padded

    def get_params(self):
        return [
            (self.W, self.dW, f"{self.name}_W"),
            (self.b, self.db, f"{self.name}_b"),
        ]

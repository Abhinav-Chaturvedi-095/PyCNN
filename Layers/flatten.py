from layers.base import Layer


class Flatten(Layer):
    def forward(self, inputs, training=True):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

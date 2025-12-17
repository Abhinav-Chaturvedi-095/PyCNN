from layers.base import Layer


class Flatten(Layer):
    def forward(self, X, training=True):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)


    def backward(self, dY):
        return dY.reshape(self.shape)
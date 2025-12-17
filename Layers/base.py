class Layer:
    def forward(self, X, training=True):
        raise NotImplementedError


    def backward(self, dY):
        raise NotImplementedError


    def params(self):
        return []


    def grads(self):
        return []
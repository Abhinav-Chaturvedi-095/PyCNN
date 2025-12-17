class Layer:
    def forward(self, inputs, training=True):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        """
        Returns list of tuples:
        (param, grad, param_name)
        """
        return []

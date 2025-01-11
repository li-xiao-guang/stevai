import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = None

    def backward(self, grad: 'Tensor' = None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.backward_fn is not None:
            self.backward_fn(grad)

    def __add__(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad)
                if other.requires_grad:
                    other.backward(grad)

            tensor = Tensor(self.data + other.data, requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data + other.data)

    def __sub__(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad)
                if other.requires_grad:
                    other.backward(grad.__neg__())

            tensor = Tensor(self.data - other.data, requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data - other.data)

    def __mul__(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(Tensor(grad.data * other.data))
                if other.requires_grad:
                    other.backward(Tensor(grad.data * self.data))

            tensor = Tensor(self.data * other.data, requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data * other.data)

    def __neg__(self):
        if self.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad.__neg__())

            tensor = Tensor(self.data * -1, requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data * -1)

    def transpose(self):
        if self.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad.transpose())

            tensor = Tensor(self.data.T, requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.T)

    def mm(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(Tensor(np.dot(grad.data, other.data.T)))
                if other.requires_grad:
                    other.backward(Tensor(np.dot(self.data.T, grad.data)))

            tensor = Tensor(np.dot(self.data, other.data), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(np.dot(self.data, other.data))

    def sum(self, axis=None):
        if self.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    if axis is None:
                        grad_broadcast = grad.data * np.ones_like(self.data)
                    else:
                        grad_expanded = np.expand_dims(grad.data, axis)
                        grad_broadcast = np.broadcast_to(grad_expanded, self.data.shape)
                    self.backward(Tensor(grad_broadcast))

            tensor = Tensor(np.sum(self.data, axis=axis), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(np.sum(self.data, axis=axis))

    def shape(self):
        return self.data.shape

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class Linear:

    def __init__(self, input_size, output_size):
        self.weights = list()

        weight = np.random.rand(input_size, output_size)
        self.weight = Tensor(weight, requires_grad=True)
        self.weights.append(self.weight)

    def forward(self, inputs):
        return inputs.mm(self.weight)


class Sequential:

    def __init__(self, layers=None):
        if layers is None:
            layers = list()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


class Loss:

    @staticmethod
    def forward(outputs, labels):
        return ((outputs - labels) * (outputs - labels)).sum(0)


class Stochastic:

    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha

    def step(self, zero=True):
        for m in self.layers:
            for w in m.weights:
                w.data -= w.grad.data * self.alpha
                if zero:
                    w.grad.data *= 0


if __name__ == '__main__':
    np.random.seed(0)

    inputs = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=True)
    labels = Tensor(np.array([[0], [1], [0], [1]]), requires_grad=True)

    models = Sequential([Linear(2, 3),
                         Linear(3, 1)])

    loss = Loss()
    optimizer = Stochastic(models.layers, alpha=0.1)

    for i in range(10):
        outputs = models.forward(inputs)

        error = loss.forward(outputs, labels)
        error.backward(Tensor(np.ones_like(error.data)))

        optimizer.step()

        print(error)

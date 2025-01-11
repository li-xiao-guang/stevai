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


if __name__ == '__main__':
    np.random.seed(0)

    inputs = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=True)
    labels = Tensor(np.array([[0], [1], [0], [1]]), requires_grad=True)

    weights = list()
    weights.append(Tensor(np.random.rand(2, 3), requires_grad=True))
    weights.append(Tensor(np.random.rand(3, 1), requires_grad=True))

    for i in range(10):
        outputs = inputs.mm(weights[0]).mm(weights[1])

        error = ((outputs - labels) * (outputs - labels)).sum(0)
        error.backward(Tensor(np.ones_like(error.data)))

        for w in weights:
            w.data -= w.grad.data * 0.1
            w.grad.data *= 0

        print(error)

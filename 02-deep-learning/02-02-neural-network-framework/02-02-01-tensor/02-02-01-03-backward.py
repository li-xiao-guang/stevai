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

            tensor = Tensor(self.data.__add__(other.data), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.__add__(other.data))

    def __sub__(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad)
                if other.requires_grad:
                    other.backward(grad.__neg__())

            tensor = Tensor(self.data.__sub__(other.data), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.__sub__(other.data))

    def __mul__(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(Tensor(grad.data * other.data))
                if other.requires_grad:
                    other.backward(Tensor(grad.data * self.data))

            tensor = Tensor(self.data.__mul__(other.data), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.__mul__(other.data))

    def __neg__(self):
        if self.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad.__neg__())

            tensor = Tensor(self.data.__neg__(), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.__neg__())

    def transpose(self):
        if self.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(grad.transpose())

            tensor = Tensor(self.data.transpose(), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.transpose())

    def dot(self, other: 'Tensor'):
        if self.requires_grad or other.requires_grad:
            def backward_fn(grad):
                if self.requires_grad:
                    self.backward(Tensor(grad.data.dot(other.data.transpose())))
                if other.requires_grad:
                    other.backward(Tensor(self.data.transpose().dot(grad.data)))

            tensor = Tensor(self.data.dot(other.data), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.dot(other.data))

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

            tensor = Tensor(self.data.sum(axis=axis), requires_grad=True)
            tensor.backward_fn = backward_fn
            return tensor

        return Tensor(self.data.sum(axis=axis))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    np.random.seed(0)

    examples = Tensor(np.array([[25.5, 65.0, 800],
                                [18.2, 45.0, 400],
                                [32.1, 75.0, 900],
                                [22.3, 62.0, 750],
                                [35.0, 80.0, 950],
                                [20.1, 55.0, 600],
                                [28.4, 70.0, 850]]),
                      requires_grad=True)
    labels = Tensor(np.array([[0.9, 0.4],
                              [0.3, 0.2],
                              [0.4, 0.5],
                              [0.8, 0.3],
                              [0.2, 0.5],
                              [0.6, 0.3],
                              [0.7, 0.4]]),
                    requires_grad=True)

    weights = list()
    weights.append(Tensor(np.random.rand(3, 8), requires_grad=True))
    weights.append(Tensor(np.random.rand(8, 2), requires_grad=True))

    for i in range(10):
        predictions = examples.dot(weights[0]).dot(weights[1])

        error = ((predictions - labels) * (predictions - labels)).sum(0)
        error.backward()

        for w in weights:
            w.data -= w.grad.data * 0.1
            w.grad.data *= 0

        print(error)

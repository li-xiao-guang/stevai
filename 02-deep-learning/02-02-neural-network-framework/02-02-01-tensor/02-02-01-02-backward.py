import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)
        self.grad = None
        self.backward_fn = None

    def backward(self, grad: 'Tensor' = None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.backward_fn is not None:
            self.backward_fn(grad)

    def __add__(self, other: 'Tensor'):
        def backward_fn(grad):
            self.backward(grad)
            other.backward(grad)

        tensor = Tensor(self.data + other.data)
        tensor.backward_fn = backward_fn
        return tensor

    def shape(self):
        return self.data.shape

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    a = Tensor([1, 2, 3])
    b = Tensor([2, 5, 8])
    c = Tensor([10, 20, 30])
    d = a + b
    e = b + c
    f = d + e
    f.backward()
    print(f)
    print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad)

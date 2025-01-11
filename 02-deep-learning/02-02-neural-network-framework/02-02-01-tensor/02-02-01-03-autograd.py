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

    def shape(self):
        return self.data.shape

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([2, 5, 8], requires_grad=True)
    c = Tensor([10, 20, 30], requires_grad=False)
    d = a + b
    e = b + c
    f = d + e
    f.backward()
    print(f)
    print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad)

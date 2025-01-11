import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other: 'Tensor'):
        tensor = Tensor(self.data + other.data)
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
    print(d, e, f)

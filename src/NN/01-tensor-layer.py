import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


class Layer:

    def __init__(self, w: Tensor):
        self.weight = w

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T))


# layer definition (out_size, in_size)
layer = Layer(Tensor(np.array([[0.5, 0.5, 0.5],
                               [1.0, 1.0, 1.0]])))

# input
example = Tensor(np.array([25.5, 65.0, 800]))

# output
prediction = layer.forward(example)
print("Prediction: ", prediction.data)

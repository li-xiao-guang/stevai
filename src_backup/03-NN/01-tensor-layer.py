import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


class Layer:

    def __init__(self, w: Tensor):
        self.weight = w

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T))


# input
example = Tensor([25.5, 65.0, 800])

# layer definition (out_size, in_size)
layer = Layer(Tensor([[0.5, 0.5, 0.5],
                      [1.0, 1.0, 1.0]]))

# output
prediction = layer(example)
print(f'Prediction: {prediction.data}')

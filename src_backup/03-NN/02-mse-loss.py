import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


class Layer:

    def __init__(self, weight: Tensor):
        self.weight = weight

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T))


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        delta = p.data - y.data
        return Tensor((delta ** 2).mean())


# input
example = Tensor([25.5, 65.0, 800])
label = Tensor([2.0, 8.0])

# layer definition (out_size, in_size)
layer = Layer(Tensor([[0.5, 0.5, 0.5],
                      [1.0, 1.0, 1.0]]))

loss = MSELoss()

# output
prediction = layer(example)
print(f'Prediction: {prediction.data}')

# evaluate
error = loss(prediction, label)
print(f'Mean Squared Error: {error.data: .4f}')

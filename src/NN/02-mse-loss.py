import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


class Layer:

    def __init__(self, weight: Tensor):
        self.weight = weight

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T))


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        delta = p.data - y.data
        mse = Tensor((delta ** 2).mean())
        return mse


# layer definition (out_size, in_size)
layer = Layer(Tensor(np.array([[0.5, 0.5, 0.5],
                               [1.0, 1.0, 1.0]])))

loss = MSELoss()

# input
example = Tensor(np.array([25.5, 65.0, 800]))
label = Tensor(np.array([2.0, 8.0]))

# output
prediction = layer.forward(example)
print("Prediction: ", prediction.data)

# evaluate
error = loss(prediction, label)
print("Mean Squared Error: ", f'{error.data: .4f}')

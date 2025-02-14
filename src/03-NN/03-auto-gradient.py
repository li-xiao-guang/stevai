import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = lambda: None
        self.parents = set()

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        self.backward_fn()

        for p in self.parents:
            if p.requires_grad:
                p.backward(p.grad)


class Layer:

    def __init__(self, weight: Tensor):
        self.weight = weight

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T), requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                self.weight.grad = p.grad.T.dot(x.data)
            if x.requires_grad:
                x.grad = self.weight.data.dot(p.grad.T)

        p.backward_fn = backward_fn
        p.parents = {self.weight, x}
        return p


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        delta = p.data - y.data
        mse = Tensor((delta ** 2).mean(), requires_grad=True)

        def backward_fn():
            if p.requires_grad:
                p.grad = delta

        mse.backward_fn = backward_fn
        mse.parents = {p}
        return mse


# input
example = Tensor([[25.5, 65.0, 800]])
label = Tensor([[2.0, 8.0]])

# layer definition (out_size, in_size)
layer = Layer(Tensor([[0.5, 0.5, 0.5],
                      [1.0, 1.0, 1.0]], requires_grad=True))

loss = MSELoss()

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    # output
    prediction = layer(example)
    print(f'Prediction: {prediction.data}')

    # evaluate
    error = loss(prediction, label)
    print(f'Mean Squared Error: {error.data: .4f}')

    # train
    layer.weight.grad = None
    error.backward()
    layer.weight.data -= layer.weight.grad

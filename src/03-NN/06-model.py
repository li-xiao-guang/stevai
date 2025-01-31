import numpy as np

np.random.seed(1)


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


class Linear(Layer):

    def __init__(self, in_size, out_size):
        weight = Tensor(np.ones([out_size, in_size]) / in_size, requires_grad=True)
        super().__init__(weight)

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T), requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                self.weight.grad = p.grad.T.dot(x.data)
            if x.requires_grad:
                x.grad = p.grad.dot(self.weight.data)

        p.backward_fn = backward_fn
        p.parents = {self.weight, x}
        return p


class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def weights(self):
        return [l.weight for l in self.layers]


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


class SGD:

    def __init__(self, weights, alpha=0.01):
        self.weights = weights
        self.alpha = alpha

    def zero_grad(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.grad = None

    def step(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.data -= self.alpha * w.grad


# data normalization (min-max)
def normalize(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min)


# layer definition (out_size, in_size)
model = Model([Linear(3, 8),
               Linear(8, 2)])

loss = MSELoss()
optimizer = SGD(model.weights(), alpha=0.01)

# input
examples = normalize(np.array([[25.5, 65.0, 800],
                               [18.2, 45.0, 400],
                               [32.1, 75.0, 900],
                               [22.3, 62.0, 750],
                               [35.0, 80.0, 950],
                               [20.1, 55.0, 600],
                               [28.4, 70.0, 850]]))
labels = np.array([[0.9, 0.4],
                   [0.3, 0.2],
                   [0.4, 0.5],
                   [0.8, 0.3],
                   [0.2, 0.5],
                   [0.6, 0.3],
                   [0.7, 0.4]])

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input
        example = Tensor(examples[i: i + 1])
        label = Tensor(labels[i: i + 1])

        # output
        prediction = model(example)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.data

        # train
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f'Mean Squared Error: {epoch_error / len(examples): .4f}')

# test
test_examples = normalize(np.array([[15.5, 40.0, 300],
                                    [30.2, 72.0, 880],
                                    [23.8, 68.0, 820]]))
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_predictions = model(Tensor(test_examples))

test_error = loss(test_predictions, Tensor(test_labels))
print(f'Test Mean Squared Error: {test_error.data: .4f}')

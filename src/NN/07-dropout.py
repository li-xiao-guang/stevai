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

    def __init__(self, weight: Tensor = None):
        self.weight = weight
        self.training = True

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

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Layer):

    def __init__(self, in_size, out_size):
        weight = Tensor(np.ones([out_size, in_size]), requires_grad=True)
        super().__init__(weight)


class Dropout(Layer):

    def forward(self, x: Tensor):
        if not self.training:
            return x

        mask = np.random.randint(2, size=x.data.shape)
        p = Tensor(x.data * mask, requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * mask

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Sequential:

    def __init__(self, layers):
        self.layers = layers
        self.training = True

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def weights(self):
        return [l.weight for l in self.layers]

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()


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

    def step(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.data -= self.alpha * w.grad

    def zero_grad(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.grad = None


# data normalization
def normalize(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


np.random.seed(1)

# layer definition (out_size, in_size)
layer = Sequential([Linear(3, 8),
                    Dropout(),
                    Linear(8, 2)])

loss = MSELoss()
optimizer = SGD(layer.weights(), alpha=0.01)

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
        prediction = layer.forward(example)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.data

        # train
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print("Mean Squared Error: ", f'{epoch_error / len(examples): .4f}')

# test
layer.eval()
test_examples = normalize(np.array([[15.5, 40.0, 300],
                                    [30.2, 72.0, 880],
                                    [23.8, 68.0, 820]]))
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_predictions = layer.forward(Tensor(test_examples))

test_error = loss(test_predictions, Tensor(test_labels))
print("Test Mean Squared Error: ", f'{test_error.data: .4f}')

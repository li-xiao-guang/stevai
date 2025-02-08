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

    def __init__(self, weight: Tensor = None):
        self.weight = weight
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


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


class Dropout(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

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


class Tanh(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * (1 - p.data ** 2)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Softmax(Layer):

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        temp = np.exp(x.data)
        p = Tensor(temp / temp.sum(axis=self.axis, keepdims=True), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                grad = p.grad - (p.grad * p.data).sum(axis=self.axis, keepdims=True)
                x.grad = p.data * grad

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Model:

    def __init__(self, layers):
        self.layers = layers
        self.training = True

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

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

    def zero_grad(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.grad = None

    def step(self):
        for w in self.weights:
            if w is not None and w.grad is not None:
                w.data -= self.alpha * w.grad

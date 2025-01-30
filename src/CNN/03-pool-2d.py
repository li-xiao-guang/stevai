import numpy as np

import src.CNN.nn as nn


class Convolution2D(nn.Layer):

    def __init__(self, k_rows, k_cols, k_num):
        self.k_rows = k_rows
        self.k_cols = k_cols
        self.k_num = k_num
        weight = nn.Tensor(np.ones([k_num, k_rows * k_cols]), requires_grad=True)
        super().__init__(weight)

    def forward(self, x: nn.Tensor):
        w = x.data.shape[1] - self.k_rows + 1
        h = x.data.shape[2] - self.k_cols + 1
        kernels = []
        for row in range(w):
            for col in range(h):
                k = x.data[:, row:row + self.k_rows, col:col + self.k_cols]
                kernels.append(k.reshape(-1))
        kernels = np.array(kernels)
        p = nn.Tensor(kernels.dot(self.weight.data.T).reshape((1, w, h, self.k_num)), requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                self.weight.grad = p.grad.reshape(-1, self.k_num).T.dot(kernels)

        p.backward_fn = backward_fn
        p.parents = {self.weight}
        return p


class Pool2D(nn.Layer):

    def __init__(self, p_size):
        self.p_size = p_size
        super().__init__()

    def forward(self, x: nn.Tensor):
        w = x.data.shape[1] // self.p_size
        h = x.data.shape[2] // self.p_size
        kernels = []
        for row in range(w):
            for col in range(h):
                r = row * self.p_size
                c = col * self.p_size
                k = x.data[:, r:r + self.p_size, c:c + self.p_size]
                kernels.append(k.reshape(1, self.p_size * self.p_size, -1).max(1))
        kernels = np.array(kernels).reshape((1, w, h, -1))
        p = nn.Tensor(kernels, requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = np.repeat(p.grad, self.p_size, axis=1)
                x.grad = np.repeat(x.grad, self.p_size, axis=2)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Flatten(nn.Layer):

    def forward(self, x: nn.Tensor):
        p = nn.Tensor(np.array([x.data.flatten()]), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad.reshape(x.data.shape)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# normalization
def normalize(x, y):
    inputs = x / 255
    targets = np.zeros((len(y), output_num))
    targets[range(len(y)), y] = 1
    return inputs, targets


np.random.seed(1)
input_rows, input_cols = (28, 28)
kernel_rows, kernel_cols = (3, 3)
kernel_num = 16
pool_size = 2
output_num = 10

sample_num = 1000

flatten_size = (input_rows - kernel_rows + 1) // pool_size * (input_cols - kernel_cols + 1) // pool_size * kernel_num

# layer definition
model = nn.Model([Convolution2D(kernel_rows, kernel_cols, kernel_num),
                  Pool2D(pool_size),
                  Flatten(),
                  nn.Tanh(),
                  nn.Linear(flatten_size, 64),
                  nn.Tanh(),
                  nn.Dropout(),
                  nn.Linear(64, output_num),
                  nn.Softmax(1)])

loss = nn.MSELoss()
optimizer = nn.SGD(model.weights(), alpha=0.01)

# inputs
with np.load('mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'][:sample_num], f['y_train'][:sample_num]
    x_test, y_test = f['x_test'][:sample_num], f['y_test'][:sample_num]

examples, labels = normalize(x_train, y_train)

# epochs
epoch_num = 30
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input
        example = nn.Tensor(examples[i: i + 1])
        label = nn.Tensor(labels[i: i + 1])

        # output
        prediction = model.forward(example)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.data

        # train
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print("Mean Squared Error: ", f'{epoch_error / len(examples): .4f}')

# test
model.eval()
test_result = 0

test_examples, test_labels = normalize(x_test, y_test)

for i in range(len(test_examples)):
    test_example = nn.Tensor(test_examples[i: i + 1])
    test_label = nn.Tensor(test_labels[i: i + 1])

    test_predictions = model.forward(test_example)
    if test_predictions.data.argmax() == test_label.data.argmax():
        test_result += 1

print("Test Mean Squared Error:", test_result, "of", len(test_examples))
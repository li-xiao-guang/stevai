import numpy as np

import nn


class Flatten(nn.Layer):

    def __call__(self, x: nn.Tensor):
        return self.forward(x)

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


input_rows, input_cols = (28, 28)
output_num = 10

sample_num = 1000

flatten_size = input_rows * input_cols

# layer definition
model = nn.Model([Flatten(),
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
epoch_num = 20
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input
        example = nn.Tensor(examples[i: i + 1])
        label = nn.Tensor(labels[i: i + 1])

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
model.eval()
test_result = 0

test_examples, test_labels = normalize(x_test, y_test)

for i in range(len(test_examples)):
    test_example = nn.Tensor(test_examples[i: i + 1])
    test_label = nn.Tensor(test_labels[i: i + 1])

    test_predictions = model(test_example)
    if test_predictions.data.argmax() == test_label.data.argmax():
        test_result += 1

print(f'Test Mean Squared Error: {test_result} of {len(test_examples)}')

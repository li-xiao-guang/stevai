import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# data normalization
def normalize(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


# layer definition (out_size, in_size)
hidden = nn.Linear(3, 8, bias=False)
hidden.weight = nn.Parameter(torch.ones_like(hidden.weight) / 3)
output = nn.Linear(8, 2, bias=False)
output.weight = nn.Parameter(torch.ones_like(output.weight) / 8)
model = nn.Sequential(hidden, output)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
        example = torch.Tensor(examples[i: i + 1])
        label = torch.Tensor(labels[i: i + 1])

        # output
        prediction = model(example)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.data

        # train
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print("Mean Squared Error: ", f'{epoch_error / len(examples): .4f}')

# test
test_examples = normalize(np.array([[15.5, 40.0, 300],
                                    [30.2, 72.0, 880],
                                    [23.8, 68.0, 820]]))
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_predictions = model(torch.Tensor(test_examples))

test_error = loss(test_predictions, torch.Tensor(test_labels))
print("Test Mean Squared Error: ", f'{test_error.data: .4f}')

import numpy as np
import torch
import torch.nn as nn

np.random.seed(1)
torch.set_default_dtype(torch.float64)


class Dropout(nn.Module):

    def forward(self, x):
        if not self.training:
            return x

        mask = torch.Tensor(np.random.randint(2, size=x.shape))
        return x * mask


# data normalization (min-max)
def normalize(x):
    x_min = torch.min(x, dim=0).values
    x_max = torch.max(x, dim=0).values
    return (x - x_min) / (x_max - x_min)


# layer definition (out_size, in_size)
hidden = nn.Linear(3, 8, bias=False)
hidden.weight = nn.Parameter(torch.ones_like(hidden.weight) / 3)
output = nn.Linear(8, 2, bias=False)
output.weight = nn.Parameter(torch.ones_like(output.weight) / 8)
model = nn.Sequential(hidden, nn.Tanh(), Dropout(), output, nn.Softmax(1))

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# input
examples = normalize(torch.Tensor([[25.5, 65.0, 800],
                                   [18.2, 45.0, 400],
                                   [32.1, 75.0, 900],
                                   [22.3, 62.0, 750],
                                   [35.0, 80.0, 950],
                                   [20.1, 55.0, 600],
                                   [28.4, 70.0, 850]]))
labels = torch.Tensor([[0.9, 0.4],
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
        example = examples[i: i + 1]
        label = labels[i: i + 1]

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
test_examples = normalize(torch.Tensor([[15.5, 40.0, 300],
                                        [30.2, 72.0, 880],
                                        [23.8, 68.0, 820]]))
test_labels = torch.Tensor([[0.4, 0.3],
                            [0.5, 0.4],
                            [0.9, 0.4]])

model.eval()
test_predictions = model(test_examples)

test_error = loss(test_predictions, test_labels)
print(f'Test Mean Squared Error: {test_error.data: .4f}')

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# layer definition (out_size, in_size)
weight = np.array([[0.5, 0.5, 0.5],
                   [1.0, 1.0, 1.0]])

layer = nn.Linear(3, 2, bias=False)
layer.weight = nn.Parameter(torch.Tensor(weight))

loss = nn.MSELoss()
optimizer = torch.optim.SGD(layer.parameters(), lr=0.000001)

# input
example = torch.Tensor(np.array([[25.5, 65.0, 800]]))
label = torch.Tensor(np.array([[2.0, 8.0]]))

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    # output
    prediction = layer(example)
    print("Prediction: ", prediction.data)

    # evaluate
    error = loss(prediction, label)
    print("Mean Squared Error: ", f'{error.data: .4f}')

    # train
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

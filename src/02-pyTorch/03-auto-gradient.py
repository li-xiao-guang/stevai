import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# input
example = torch.Tensor([25.5, 65.0, 800])
label = torch.Tensor([2.0, 8.0])

# layer definition (out_size, in_size)
weight = torch.Tensor([[0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0]])

layer = nn.Linear(3, 2, bias=False)
layer.weight = nn.Parameter(weight)

loss = nn.MSELoss()

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

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# input
example = torch.Tensor([25.5, 65.0, 800])

# layer definition (out_size, in_size)
weight = torch.Tensor([[0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0]])

layer = nn.Linear(3, 2, bias=False)
layer.weight = nn.Parameter(weight)

# output
prediction = layer(example)
print(f'Prediction: {prediction.data}')

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 模型推理函数
model = nn.Linear(2, 1)

# 模型参数（权重，偏差）
model.weight = nn.Parameter(torch.Tensor([[1.0, 1.0]]))
model.bias = nn.Parameter(torch.Tensor([0.5]))

# 观测数据（温度，湿度）
obs = torch.Tensor([25.3, 65.0])

# 模型推理
prediction = model(obs)
print(f'销量预测：{prediction.data[0]:.2f}')

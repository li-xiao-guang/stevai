import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 模型推理函数
model = nn.Linear(2, 1)

# 模型参数（权重，偏差）
model.weight = nn.Parameter(torch.Tensor([[1.0, 1.0]]))
model.bias = nn.Parameter(torch.Tensor([0.5]))

# 损失函数
loss = nn.MSELoss()

# 观测数据（温度，湿度）
obs = torch.Tensor([25.3, 65.0])

# 模型推理
prediction = model(obs)
print(f'销量预测：{prediction.data[0]:.2f}')

# 实际结果（冰淇淋销量）
actual = torch.Tensor([122])

# 模型评价
error = loss(prediction, actual)
print(f'误差：{error.data:.4f}')

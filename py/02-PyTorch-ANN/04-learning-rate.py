import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 学习率
ALPHA = 0.0001

# 模型推理函数
model = nn.Linear(2, 1)

# 模型参数（权重，偏差）
model.weight = nn.Parameter(torch.Tensor([[1.0, 1.0]]))
model.bias = nn.Parameter(torch.Tensor([0.5]))

# 损失函数
loss = nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)

# 特征数据
features = torch.Tensor([[28.1, 58.0], [22.5, 72.0], [31.4, 45.0], [19.8, 85.0], [27.6, 63]])
# 标签数据
labels = torch.Tensor([[165], [95], [210], [70], [155]])

# 模型训练
for i in range(len(features)):
    feature = features[i: i + 1]
    label = labels[i: i + 1]

    # 模型推理
    prediction = model(feature)
    # 计算误差
    error = loss(prediction, label)
    # 反向传播
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

    print(f"权重：{model.weight.data.numpy()}")
    print(f"偏差：{model.bias.data.numpy()}")
    print(f"误差：{error.data:.4f}\n")

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

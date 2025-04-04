import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 学习率
ALPHA = 0.00001

# 模型推理函数
hidden = nn.Linear(2, 4)
output = nn.Linear(4, 1)
model = nn.Sequential(hidden, nn.ReLU(), output)

# 损失函数（平均平方差）
loss = nn.MSELoss()

# 模型参数（权重，偏差）
hidden.weight = nn.Parameter(torch.Tensor([[1.0, 1.0], [1.0, 0.5], [0.5, 1.0], [0.5, 0.5]]))
hidden.bias = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]))
output.weight = nn.Parameter(torch.Tensor([[1.0, 1.0, 1.0, 1.0]]))
output.bias = nn.Parameter(torch.Tensor([0.5]))
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)

# 特征数据
features = torch.Tensor([[28.1, 58.0], [22.5, 72.0], [31.4, 45.0], [19.8, 85.0], [27.6, 63]])
# 标签数据
labels = torch.Tensor([[165], [95], [210], [70], [155]])

# 模型训练
epoches = 1000
for i in range(epoches):
    epoch_error = 0

    for i in range(len(features)):
        feature = features[i: i + 1]
        label = labels[i: i + 1]

        # 模型推理
        prediction = model(feature)
        # 计算误差
        error = loss(prediction, label)
        epoch_error += error
        # 反向传播
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f"隐藏层权重：{hidden.weight.data.numpy()}")
    print(f"隐藏层偏差：{hidden.bias.data.numpy()}")
    print(f"输出层权重：{output.weight.data.numpy()}")
    print(f"输出层偏差：{output.bias.data.numpy()}")
    print(f"误差：{epoch_error:.4f}\n")

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

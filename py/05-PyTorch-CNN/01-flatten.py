import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# 学习率
ALPHA = 0.01
# 图像尺寸（行，列）
ROW, COL = (28, 28)

# 模型推理函数
hidden = nn.Linear(ROW * COL, 64)
output = nn.Linear(64, 10)
model = nn.Sequential(nn.Flatten(), hidden, output)

# 模型参数（权重，偏差）
hidden.weight = nn.Parameter(torch.Tensor(np.random.random([64, ROW * COL]) / (ROW * COL)))
hidden.bias = nn.Parameter(torch.Tensor(np.random.random([64]) / (ROW * COL)))
output.weight = nn.Parameter(torch.Tensor(np.random.random([10, 64]) / 64))
output.bias = nn.Parameter(torch.Tensor(np.random.random([10]) / 64))

# 损失函数
loss = nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)


# 规范化函数
def normalize(x, y):
    inputs = x / 255
    targets = np.zeros((len(y), 10))
    targets[range(len(y)), y] = 1
    return inputs, targets


# 训练数据（特征数据，标签数据）
with np.load('mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'][:2000], f['y_train'][:2000]
features, labels = normalize(x_train, y_train)
features = torch.Tensor(features)
labels = torch.Tensor(labels)

# 模型训练
epoches = 10
for _ in range(epoches):
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

    print(f"隐藏层权重：{hidden.weight.data.numpy()}")
    print(f"隐藏层偏差：{hidden.bias.data.numpy()}")
    print(f"输出层权重：{output.weight.data.numpy()}")
    print(f"输出层偏差：{output.bias.data.numpy()}")
    print(f"误差：{error.data:.4f}\n")

# 测试数据（特征数据，标签数据）
with np.load('mnist.npz', allow_pickle=True) as f:
    x_test, y_test = f['x_test'][:1000], f['y_test'][:1000]
features, labels = normalize(x_test, y_test)
features = torch.Tensor(features)
labels = torch.Tensor(labels)

# 模型推理
result = 0
for i in range(len(features)):
    feature = features[i: i + 1]
    label = labels[i: i + 1]

    prediction = model(feature)
    if prediction.data.argmax() == label.data.argmax():
        result += 1

print(f'测试结果：{result} of {len(labels)}')

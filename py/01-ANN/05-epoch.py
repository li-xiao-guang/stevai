import numpy as np

# 学习率
ALPHA = 0.0001


# 模型推理函数
def predict(x, w, b):
    return x.dot(w.T) + b


# 损失函数（平均平方差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 梯度函数
def gradient(p, y):
    return p - y


# 反向传播函数
def backward(x, d, w, b):
    w -= d.T.dot(x) * ALPHA
    b -= np.sum(d, axis=0) * ALPHA
    return w, b


# 特征数据
features = np.array([[28.1, 58.0], [22.5, 72.0], [31.4, 45.0], [19.8, 85.0], [27.6, 63]])

# 标签数据
labels = np.array([165, 95, 210, 70, 155])

# 模型参数（权重，偏差）
weight = np.array([1.0, 1.0])
bias = 0.5

# 模型训练
epoches = 10
for i in range(epoches):
    error = 0
    for i in range(len(features)):
        feature = features[i: i + 1]
        label = labels[i: i + 1]
        # 模型推理
        prediction = predict(feature, weight, bias)
        # 计算损失
        delta = gradient(prediction, label)
        # 反向传播
        (weight, bias) = backward(feature, delta, weight, bias)
        # 计算误差
        error += mse_loss(prediction, label)
    print(f"权重：{weight}")
    print(f"偏差：{bias}")
    print(f"误差：{error:.4f}\n")

# 观测数据（温度，湿度）
obs = np.array([25.3, 65.0])

# 模型推理
prediction = predict(obs, weight, bias)
print(f'销量预测：{prediction:.2f}')

# 实际结果（冰淇淋销量）
actual = 122

# 模型评价
error = mse_loss(prediction, actual)
print(f'误差：{error:.4f}')

import numpy as np

# 学习率
ALPHA = 0.00001


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


# activation function
def relu(x):
    return np.maximum(0, x)


# activation function backward
def relu_backward(y, d):
    return (y > 0).astype(float) * d


# 特征数据
features = np.array([[28.1, 58.0], [22.5, 72.0], [31.4, 45.0], [19.8, 85.0], [27.6, 63]])

# 标签数据
labels = np.array([165, 95, 210, 70, 155])

# 隐藏层模型参数（权重，偏差）
hidden_weight = np.array([[1.0, 1.0], [1.0, 0.5], [0.5, 1.0], [0.5, 0.5]])
hidden_bias = [0.5, 0.5, 0.5, 0.5]
# 输出层模型参数（权重，偏差）
output_weight = np.array([[1.0, 1.0, 1.0, 1.0]])
output_bias = [0.5]

# 模型训练
epoches = 1000
for i in range(epoches):
    error = 0
    for i in range(len(features)):
        feature = features[i: i + 1]
        label = labels[i: i + 1]
        # 模型推理
        hidden = relu(predict(feature, hidden_weight, hidden_bias))
        prediction = predict(hidden, output_weight, output_bias)
        # 计算损失
        output_delta = gradient(prediction, label)
        hidden_delta = output_delta.dot(output_weight)
        hidden_delta = relu_backward(hidden, hidden_delta)
        # 反向传播
        (output_weight, output_bias) = backward(hidden, output_delta, output_weight, output_bias)
        (hidden_weight, hidden_bias) = backward(feature, hidden_delta, hidden_weight, hidden_bias)
        # 计算误差
        error += mse_loss(prediction, label)
    print(f"隐藏层权重：{hidden_weight}")
    print(f"隐藏层偏差：{hidden_bias}")
    print(f"输出层权重：{output_weight}")
    print(f"输出层偏差：{output_bias}")
    print(f"误差：{error:.4f}\n")

# 观测数据（温度，湿度）
obs = np.array([25.3, 65.0])

# 模型推理
hidden = relu(predict(obs, hidden_weight, hidden_bias))
prediction = predict(hidden, output_weight, output_bias)
print(f'销量预测：{prediction[0]:.2f}')

# 实际结果（冰淇淋销量）
actual = 122

# 模型评价
error = mse_loss(prediction, actual)
print(f'误差：{error:.4f}')

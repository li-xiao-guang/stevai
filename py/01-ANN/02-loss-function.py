import numpy as np


# 模型推理函数
def predict(x, w, b):
    return x.dot(w.T) + b


# 损失函数（平均平方差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 模型参数（权重，偏差）
weight = np.array([1.0, 1.0])
bias = 0.5

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

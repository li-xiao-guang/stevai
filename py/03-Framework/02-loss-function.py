import numpy as np


# 张量类
class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


# 线性回归层类
class Linear:

    def __init__(self, w: Tensor = None, b: Tensor = None):
        self.weight = w
        self.bias = b

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T) + self.bias.data)


# MSE损失函数类
class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        return Tensor(((p.data - y.data) ** 2).mean())


# 模型推理函数
model = Linear()

# 模型参数（权重，偏差）
model.weight = Tensor([[1.0, 1.0]])
model.bias = Tensor([0.5])

# 损失函数
loss = MSELoss()

# 观测数据（温度，湿度）
obs = Tensor([25.3, 65.0])

# 模型推理
prediction = model(obs)
print(f'销量预测：{prediction.data[0]:.2f}')

# 实际结果（冰淇淋销量）
actual = Tensor([122])

# 模型评价
error = loss(prediction, actual)
print(f'误差：{error.data:.4f}')

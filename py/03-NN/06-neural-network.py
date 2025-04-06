import numpy as np


# 张量类
class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = lambda: None
        self.parents = set()

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad
        self.backward_fn()

        for p in self.parents:
            if p.requires_grad:
                p.backward(p.grad)


# 线性回归层类
class Linear:

    def __init__(self, w: Tensor = None, b: Tensor = None):
        self.weight = w
        self.bias = b

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T) + self.bias.data, requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                self.weight.grad = p.grad.T.dot(x.data)
            if self.bias.requires_grad:
                self.bias.grad = np.sum(p.grad, axis=0)
            if x.requires_grad:
                x.grad = p.grad.dot(self.weight.data)

        p.backward_fn = backward_fn
        p.parents = {self.weight, x}
        return p


# 模型类
class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in (l.weight, l.bias)]


# MSE损失函数类
class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        mse = Tensor(((p.data - y.data) ** 2).mean(), requires_grad=True)

        def backward_fn():
            if p.requires_grad:
                p.grad = (p.data - y.data) * 2 / len(y.data)

        mse.backward_fn = backward_fn
        mse.parents = {p}
        return mse


# SGD优化器类
class SGD:

    def __init__(self, parameters, alpha=0.01):
        self.parameters = parameters
        self.alpha = alpha

    def zero_grad(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.grad = None

    def step(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.data -= p.grad * self.alpha


# 学习率
ALPHA = 0.00001

# 模型推理函数
hidden = Linear()
output = Linear()
model = Model([hidden, output])

# 模型参数（权重，偏差）
hidden.weight = Tensor([[1.0, 1.0], [1.0, 0.5], [0.5, 1.0], [0.5, 0.5]], requires_grad=True)
hidden.bias = Tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
output.weight = Tensor([[1.0, 1.0, 1.0, 1.0]], requires_grad=True)
output.bias = Tensor([0.5], requires_grad=True)

# 损失函数
loss = MSELoss()
# 优化器
optimizer = SGD(model.parameters(), alpha=ALPHA)

# 特征数据
features = Tensor([[28.1, 58.0], [22.5, 72.0], [31.4, 45.0], [19.8, 85.0], [27.6, 63]])
# 标签数据
labels = Tensor([[165], [95], [210], [70], [155]])

# 模型训练
epoches = 1000
for i in range(epoches):
    for i in range(len(features.data)):
        feature = Tensor(features.data[i: i + 1])
        label = Tensor(labels.data[i: i + 1])

        # 模型推理
        prediction = model(feature)
        # 计算误差
        error = loss(prediction, label)
        # 反向传播
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f"隐藏层权重：{hidden.weight.data}")
    print(f"隐藏层偏差：{hidden.bias.data}")
    print(f"输出层权重：{output.weight.data}")
    print(f"输出层偏差：{output.bias.data}")
    print(f"误差：{error.data:.4f}\n")

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

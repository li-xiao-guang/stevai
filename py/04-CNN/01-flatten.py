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


# 层基础类
class Layer:

    def parameters(self):
        return []


# 线性回归层类
class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.random.random([out_size, in_size]) / in_size, requires_grad=True)
        self.bias = Tensor(np.random.random([out_size]) / in_size, requires_grad=True)

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
        p.parents = {self.weight, self.bias, x}
        return p

    def parameters(self):
        return [self.weight, self.bias]


# 扁平化类
class Flatten(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @staticmethod
    def forward(x: Tensor):
        p = Tensor(np.array([x.data.flatten()]), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad.reshape(x.data.shape)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# ReLU激活函数类
class ReLU(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @staticmethod
    def forward(x: Tensor):
        p = Tensor(np.maximum(0, x.data), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = (p.data > 0).astype(float) * p.grad

        p.backward_fn = backward_fn
        p.parents = {x}
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
        return [p for l in self.layers for p in l.parameters()]


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
                p.data -= self.alpha * p.grad


# 学习率
ALPHA = 0.01
# 图像尺寸（行、列）
ROW, COL = (28, 28)

# 模型推理函数
hidden = Linear(ROW * COL, 64)
output = Linear(64, 10)
model = Model([Flatten(), hidden, output])

# 损失函数
loss = MSELoss()
# 优化器
optimizer = SGD(model.parameters(), alpha=ALPHA)


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

# 模型训练
epoches = 10
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

# 测试数据（特征数据，标签数据）
with np.load('mnist.npz', allow_pickle=True) as f:
    x_test, y_test = f['x_test'][:1000], f['y_test'][:1000]
features, labels = normalize(x_test, y_test)

# 模型推理
result = 0
for i in range(len(features)):
    feature = Tensor(features[i: i + 1])
    label = Tensor(labels[i: i + 1])

    prediction = model(feature)
    if prediction.data.argmax() == label.data.argmax():
        result += 1

print(f'测试结果：{result} of {len(labels)}')

import re

import numpy as np
import pandas as pd


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

    def __init__(self):
        self.training = True

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# 线性回归层类
class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.random.random([out_size, in_size]) / in_size, requires_grad=True)
        self.bias = Tensor(np.random.random([out_size]) / in_size, requires_grad=True)
        super().__init__()

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


# 扁平化层类
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


# 丢弃层类
class Dropout(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        if not self.training:
            return x

        mask = np.random.randint(2, size=x.data.shape)
        p = Tensor(x.data * mask, requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * mask

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# 二维卷积层类
class Convolution2D(Layer):

    def __init__(self, rows, cols, num):
        self.rows = rows
        self.cols = cols
        self.num = num
        self.weight = Tensor(np.random.random([num, rows * cols]), requires_grad=True)
        super().__init__()

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        w = x.data.shape[1] - self.rows + 1
        h = x.data.shape[2] - self.cols + 1
        kernels = []
        for row in range(w):
            for col in range(h):
                k = x.data[:, row:row + self.rows, col:col + self.cols]
                kernels.append(k.reshape(-1))
        kernels = np.array(kernels)
        p = Tensor(kernels.dot(self.weight.data.T).reshape((1, w, h, self.num)), requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                self.weight.grad = p.grad.reshape(-1, self.num).T.dot(kernels)

        p.backward_fn = backward_fn
        p.parents = {self.weight, x}
        return p

    def parameters(self):
        return [self.weight]


# 二维池化层类
class Pool2D(Layer):

    def __init__(self, size):
        self.size = size
        super().__init__()

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        w = x.data.shape[1] // self.size
        h = x.data.shape[2] // self.size
        kernels = []
        for row in range(w):
            for col in range(h):
                r = row * self.size
                c = col * self.size
                k = x.data[:, r:r + self.size, c:c + self.size]
                kernels.append(k.reshape(1, self.size * self.size, -1).max(1))
        kernels = np.array(kernels).reshape((1, w, h, -1))
        p = Tensor(kernels, requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = np.repeat(p.grad, self.size, axis=1)
                x.grad = np.repeat(x.grad, self.size, axis=2)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# 单热向量嵌入层类
class OneHotEmbedding(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.random.random([out_size, in_size]) / out_size, requires_grad=True)
        super().__init__()

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(np.sum(self.weight.data.T[x.data], axis=1), requires_grad=True)

        def backward_fn():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                self.weight.grad.T[x.data] = p.grad

        p.backward_fn = backward_fn
        p.parents = {self.weight}
        return p

    def parameters(self):
        return [self.weight]


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


# Tanh激活函数类
class Tanh(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @staticmethod
    def forward(x: Tensor):
        p = Tensor(np.tanh(x.data), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * (1 - p.data ** 2)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# Sigmoid激活函数类
class Sigmoid(Layer):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @staticmethod
    def forward(x: Tensor):
        p = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * p.data * (1 - p.data)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


# Softmax激活函数类
class Softmax(Layer):

    def __init__(self, axis=1):
        self.axis = axis
        super().__init__()

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        temp = np.exp(x.data)
        p = Tensor(temp / temp.sum(axis=self.axis, keepdims=True), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                grad = p.grad - (p.grad * p.data).sum(axis=self.axis, keepdims=True)
                x.grad = p.data * grad

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

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()


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

    def __init__(self, params, lr=0.01):
        self.parameters = params
        self.alpha = lr

    def zero_grad(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.grad = None

    def step(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.data -= self.alpha * p.grad


# 映射类
class Word2Index:

    def __init__(self, filename):
        df = pd.read_csv(filename)

        x = df['review']
        y = df['sentiment']
        x = x.apply(self.clean_html)
        x = x.apply(self.convert_lower)
        x = x.apply(self.remove_special)

        self.reviews = list(map(lambda s: s.split(), x))
        self.sentiments = list(y)
        self.words = list(set(w for r in self.reviews for w in r))
        self.word2index = {w: self.words.index(w) for w in self.words}

    @staticmethod
    def clean_html(text):
        p = re.compile(r'<.*?>')
        return p.sub('', text)

    @staticmethod
    def convert_lower(text):
        return text.lower()

    @staticmethod
    def remove_special(text):
        x = ''
        for t in text:
            x = x + t if t.isalnum() else x + ' '
        return x


# 单热向量映射类
class OneHotEncoding(Word2Index):

    def __init__(self, filename):
        super().__init__(filename)
        self.features = [list(set(self.word2index[w] for w in r)) for r in self.reviews]
        self.labels = [0 if s == "negative" else 1 for s in self.sentiments]


# 学习率
ALPHA = 0.01

# 加载数据（特征数据，标签数据，词汇表，映射表）
data = OneHotEncoding('imdb.csv')

# 模型推理函数
embedding = OneHotEmbedding(len(data.words), 64)
output = Linear(64, 1)
model = Model([embedding, ReLU(), output, Sigmoid()])

# 损失函数
loss = MSELoss()
# 优化器
optimizer = SGD(model.parameters(), lr=ALPHA)

# 训练数据（特征数据，标签数据）
features = data.features[:2000]
labels = data.labels[:2000]

# 模型训练
epoches = 10
for i in range(epoches):
    for i in range(len(features)):
        feature = Tensor(features[i: i + 1])
        label = Tensor(labels[i: i + 1])

        # 模型推理
        prediction = model(feature)
        # 计算误差
        error = loss(prediction, label)
        # 反向传播
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f"嵌入层权重：{embedding.weight.data}")
    print(f"输出层权重：{output.weight.data}")
    print(f"输出层偏差：{output.bias.data}")
    print(f"误差：{error.data:.4f}\n")

# 测试数据（特征数据，标签数据）
features = data.features[-1000:]
labels = data.labels[-1000:]

# 模型推理
model.eval()

result = 0
for i in range(len(features)):
    feature = Tensor(features[i: i + 1])
    label = Tensor(labels[i: i + 1])

    prediction = model(feature)
    if np.abs(prediction.data - label.data) < 0.5:
        result += 1

print(f'测试结果：{result} of {len(labels)}')

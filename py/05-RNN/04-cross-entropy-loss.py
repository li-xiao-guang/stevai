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
        p.parents = {self.weight}
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


# 词袋嵌入层类
class BagOfWordsEmbedding(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.random.random([out_size, in_size]) / in_size, requires_grad=True)
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


# 连续词袋嵌入层类
class ContinuousBagOfWordsEmbedding(Layer):

    def __init__(self, vocabs, size):
        self.matrix = Tensor(np.random.random([vocabs, size]) / size, requires_grad=True)
        self.weight = Tensor(np.random.random([size, size]) / size, requires_grad=True)
        super().__init__()

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        grads = []
        d = np.zeros(len(self.weight.data))
        for w in x.data.flatten():
            d = self.matrix.data[w] + d.dot(self.weight.data.T)
            grads.append(d)
        p = Tensor([d], requires_grad=True)

        def backward_fn():
            if self.matrix.requires_grad:
                if self.matrix.grad is None:
                    self.matrix.grad = np.zeros_like(self.matrix.data)
                for i in range(len(x.data.flatten()), 0, -1):
                    w = x.data.flatten()[i - 1];
                    self.matrix.grad[w] += grads[i - 1]
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                for i in range(len(x.data.flatten()), 0, -1):
                    self.weight.grad += p.grad.dot(grads[i - 1])

        p.backward_fn = backward_fn
        p.parents = {self.matrix, self.weight}
        return p

    def parameters(self):
        return [self.matrix, self.weight]


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


class CELoss:

    def __call__(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=1, keepdims=True))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)

        log = np.log(softmax + 1e-10)
        ce = Tensor(-np.sum(y.data * log) / len(p.data), requires_grad=True)

        def backward_fn():
            if p.requires_grad:
                p.grad = (softmax - y.data) / len(p.data)

        ce.backward_fn = backward_fn
        ce.parents = {p}
        return ce


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


# 单热映射类
class OneHotEncoding(Word2Index):

    def __init__(self, filename):
        super().__init__(filename)
        self.features = [list(set(self.word2index[w] for w in r)) for r in self.reviews]
        self.labels = [0 if s == "negative" else 1 for s in self.sentiments]


# 词袋映射类
class BagOfWordsEncoding(Word2Index):

    def __init__(self, filename):
        super().__init__(filename)
        self.features = [list(self.word2index[w] for w in r) for r in self.reviews]
        self.labels = [0 if s == "negative" else 1 for s in self.sentiments]


# 学习率
ALPHA = 0.01

# 加载数据（特征数据，标签数据，词汇表，映射表）
dataset = BagOfWordsEncoding('reviews.csv')

# 模型推理函数
embedding = ContinuousBagOfWordsEmbedding(len(dataset.words), 128)
hidden = Linear(128, 64)
output = Linear(64, len(dataset.words))
model = Model([embedding, hidden, ReLU(), output, Sigmoid()])

# 损失函数
loss = CELoss()
# 优化器
optimizer = SGD(model.parameters(), lr=ALPHA)

# 训练数据（特征数据，标签数据）
features = dataset.features[:-1]
labels = dataset.labels[:-1]

# 模型训练
epoches = 10
for _ in range(epoches):
    for i in range(len(features)):
        feature = features[i]
        for j in range(len(feature) - 2):
            feature_word = Tensor([feature[:j + 1]])
            label_word = Tensor([np.zeros(len(dataset.words))])
            label_word.data[0][feature[j + 1]] = 1

            # 模型推理
            prediction = model(feature_word)
            # 计算误差
            error = loss(prediction, label_word)
            prediction_word = prediction.data[0].argmax()
            # 反向传播
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

    print(f"嵌入层权重：{embedding.weight.data}")
    print(f"输出层权重：{output.weight.data}")
    print(f"输出层偏差：{output.bias.data}")
    print(f"误差：{error.data:.4f}\n")

# 测试数据（特征数据，标签数据）
features = dataset.features[-1:]
labels = dataset.labels[-1:]

# 模型推理
model.eval()

feature = features[0]
feature_word = Tensor([feature[:10]])
for i in range(10):
    prediction = model(feature_word)
    prediction_word = prediction.data[0].argmax()
    feature_word.data = np.array([np.append(feature_word.data[0], prediction_word)])

words = list(dataset.words[i] for i in feature_word.data[0])

print(f'生成结果：{' '.join(words)}')

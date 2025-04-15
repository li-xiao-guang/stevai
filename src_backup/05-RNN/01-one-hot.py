import re

import numpy as np
import pandas as pd

import nn


class Embedding(nn.Layer):

    def __init__(self, in_size, out_size):
        weight = nn.Tensor(np.ones([out_size, in_size]) / in_size, requires_grad=True)
        super().__init__(weight)

    def __call__(self, x: nn.Tensor):
        return self.forward(x)

    def forward(self, x: nn.Tensor):
        p = nn.Tensor(np.array([x.data.flatten()]), requires_grad=True)

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad.reshape(x.data.shape)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


def clean_html(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def convert_lower(text):
    return text.lower()


def remove_special(text):
    x = ''
    for t in text:
        if t.isalnum():
            x = x + t
        else:
            x = x + ' '
    return x


# normalization
def normalize(x, y):
    x = x.apply(clean_html)
    x = x.apply(convert_lower)
    x = x.apply(remove_special)
    reviews = list(map(lambda s: s.split(), x))
    words = list(set(word for review in reviews for word in review))
    word2index = {word: words.index(word) for word in words}
    inputs = [list(set(word2index[word] for word in review)) for review in reviews]
    targets = list(y.map({'positive': 1, 'negative': 0}))
    return len(words), inputs, targets


# inputs
sample_num = 2000

df = pd.read_csv('imdb.csv')
word_count, examples, labels = normalize(df['review'], df['sentiment'])
examples = examples[:sample_num]
labels = labels[:sample_num]

# layer definition
model = nn.Model([Embedding(word_count, 64),
                  nn.Tanh(),
                  nn.Dropout(),
                  nn.Linear(64, 1),
                  nn.Softmax(1)])

loss = nn.MSELoss()
optimizer = nn.SGD(model.weights(), alpha=0.01)

# epochs
epoch_num = 20
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input
        example = nn.Tensor(examples[i: i + 1])
        label = nn.Tensor(labels[i: i + 1])

        # output
        prediction = model(example)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.data

        # train
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f'Mean Squared Error: {epoch_error / len(examples): .4f}')

# test
test_sample_num = 1000

with np.load('mnist.npz', allow_pickle=True) as f:
    x_test, y_test = f['x_test'][:test_sample_num], f['y_test'][:test_sample_num]

word_count, test_examples, test_labels = normalize(x_test, y_test)

model.eval()
test_result = 0

for i in range(len(test_examples)):
    test_example = nn.Tensor(test_examples[i: i + 1])
    test_label = nn.Tensor(test_labels[i: i + 1])

    test_predictions = model(test_example)
    if test_predictions.data.argmax() == test_label.data.argmax():
        test_result += 1

print(f'Test Mean Squared Error: {test_result} of {len(test_examples)}')

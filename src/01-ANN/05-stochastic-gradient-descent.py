import numpy as np

# learning rate
alpha = 0.01


# neuron definition
def predict(x, w):
    return x.dot(w.T)


# loss function (mean squared error)
def loss(p, y):
    return ((p - y) ** 2).mean()


# gradient descent
def gradient(p, y):
    return p - y


# back propagation
def backward(x, d):
    return d.T.dot(x) * alpha


# data normalization (min-max)
def normalize(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min)


# layer definition (out_size, in_size)
weight = np.ones((2, 3)) / 3

# input
examples = normalize(np.array([[25.5, 65.0, 800],
                               [18.2, 45.0, 400],
                               [32.1, 75.0, 900],
                               [22.3, 62.0, 750],
                               [35.0, 80.0, 950],
                               [20.1, 55.0, 600],
                               [28.4, 70.0, 850]]))
labels = np.array([[0.9, 0.4],
                   [0.3, 0.2],
                   [0.4, 0.5],
                   [0.8, 0.3],
                   [0.2, 0.5],
                   [0.6, 0.3],
                   [0.7, 0.4]])

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input
        example = examples[i: i + 1]
        label = labels[i: i + 1]

        # output
        prediction = predict(example, weight)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error

        # train
        delta = gradient(prediction, label)
        weight -= backward(example, delta)

    print(f'Mean Squared Error: {epoch_error / len(examples): .4f}')

# test
test_examples = normalize(np.array([[15.5, 40.0, 300],
                                    [30.2, 72.0, 880],
                                    [23.8, 68.0, 820]]))
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_predictions = predict(test_examples, weight)

test_error = loss(test_predictions, test_labels)
print(f'Test Mean Squared Error: {test_error: .4f}')

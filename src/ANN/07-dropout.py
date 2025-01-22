import numpy as np

np.random.seed(1)

# learning rate
alpha = 0.01


# neuron definition
def predict(x, w):
    return x.dot(w.T)


# gradient descent
def backward(x, d):
    return d.T.dot(x) * alpha


# mean squared error
def loss(p, y):
    return ((p - y) ** 2).mean()


# gradient descent
def gradient(p, y):
    return p - y


# data normalization
def normalize(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


# layer definition (out_size, in_size)
hidden_weight = np.ones((8, 3))
output_weight = np.ones((2, 8))

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

        # hidden
        hidden = predict(example, hidden_weight)
        mask = np.random.randint(2, size=hidden.shape)
        hidden *= mask
        # output
        prediction = predict(hidden, output_weight)

        # evaluate
        error = loss(prediction, label)
        epoch_error += error

        # train
        output_delta = gradient(prediction, label)
        hidden_delta = output_delta.dot(output_weight)
        hidden_delta *= mask
        output_weight -= backward(hidden, output_delta)
        hidden_weight -= backward(example, hidden_delta)

    print("Mean Squared Error: ", f'{epoch_error / len(examples): .4f}')

# test
test_examples = normalize(np.array([[15.5, 40.0, 300],
                                    [30.2, 72.0, 880],
                                    [23.8, 68.0, 820]]))
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_hiddens = predict(test_examples, hidden_weight)
test_predictions = predict(test_hiddens, output_weight)

test_error = loss(test_predictions, test_labels)
print("Test Mean Squared Error: ", f'{test_error: .4f}')

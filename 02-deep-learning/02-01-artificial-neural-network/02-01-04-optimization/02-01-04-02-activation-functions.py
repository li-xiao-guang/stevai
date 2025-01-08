import numpy as np

np.random.seed(1)

# learning rate
alpha = 0.001

# activation functions
tanh = lambda x: np.tanh(x)
tanh_derivative = lambda x: 1 - np.tanh(x) ** 2
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# neuron definition
def forward(x, w):
    return np.dot(x, w)


# gradient descent
def backward(x, d):
    return alpha * np.dot(x.T, d)


# loss function
def loss(p, y):
    return np.mean((p - y) ** 2)


# derivative of the loss function
def gradient(p, y):
    return 2 * (p - y)


# normalization
def normalize(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


# layer connection definition (input_size, output_size)
input_hidden_weights = np.ones((3, 8))
hidden_output_weights = np.ones((8, 2))

# inputs
examples = np.array([[25.5, 65.0, 800],
                     [18.2, 45.0, 400],
                     [32.1, 75.0, 900],
                     [22.3, 62.0, 750],
                     [35.0, 80.0, 950],
                     [20.1, 55.0, 600],
                     [28.4, 70.0, 850]])
labels = np.array([[0.9, 0.4],
                   [0.3, 0.2],
                   [0.4, 0.5],
                   [0.8, 0.3],
                   [0.2, 0.5],
                   [0.6, 0.3],
                   [0.7, 0.4]])

examples = normalize(examples)

# epochs
epoch_num = 10
for epoch in range(epoch_num):
    epoch_error = 0

    # iteration
    for i in range(len(examples)):
        # input layer
        example = examples[i: i + 1]
        label = labels[i: i + 1]

        # hidden layer
        hidden = tanh(forward(example, input_hidden_weights))
        mask = np.random.randint(2, size=hidden.shape)
        hidden *= mask * 2

        # output layer
        prediction = softmax(forward(hidden, hidden_output_weights))

        # evaluate
        error = loss(prediction, label)
        epoch_error += error.mean()

        # train
        delta = gradient(prediction, label)
        hidden_output_weights -= backward(hidden, delta)
        hidden_delta = np.dot(delta, hidden_output_weights.T)
        hidden_delta *= mask
        hidden_delta *= tanh_derivative(hidden)
        input_hidden_weights -= backward(example, hidden_delta)

    print("Mean Squared Error: ", f'{epoch_error / len(examples): .4f}')

# test
test_examples = np.array([[15.5, 40.0, 300],
                          [30.2, 72.0, 880],
                          [23.8, 68.0, 820]])
test_labels = np.array([[0.4, 0.3],
                        [0.5, 0.4],
                        [0.9, 0.4]])

test_examples = normalize(test_examples)
test_hiddens = tanh(forward(test_examples, input_hidden_weights))
test_predictions = softmax(forward(test_hiddens, hidden_output_weights))

test_error = loss(test_predictions, test_labels)
print("Test Mean Squared Error: ", f'{test_error.mean(): .4f}')

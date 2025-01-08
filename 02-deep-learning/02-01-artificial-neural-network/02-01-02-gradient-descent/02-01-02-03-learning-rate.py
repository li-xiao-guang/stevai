import numpy as np

# learning rate
alpha = 0.01


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


# layer connection definition (input_size, output_size)
weights = [0.5, 1.0] * np.ones((3, 2))

# inputs
examples = np.array([[2.0, 5.0, 1.0]])
labels = np.array([[2.0, 8.0]])

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    # outputs
    predictions = forward(examples, weights)

    # evaluate
    error = loss(predictions, labels)
    print("Mean Squared Error: ", f'{error.mean(): .4f}')

    # train
    delta = gradient(predictions, labels)
    weights -= backward(examples, delta)

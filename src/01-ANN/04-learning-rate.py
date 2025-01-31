import numpy as np

# learning rate
alpha = 0.000001


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


# layer definition (out_size, in_size)
weight = np.array([[0.5, 0.5, 0.5],
                   [1.0, 1.0, 1.0]])

# input
example = np.array([[25.5, 65.0, 800]])
label = np.array([[2.0, 8.0]])

# epochs
epoch_num = 5
for epoch in range(epoch_num):
    # output
    prediction = predict(example, weight)
    print(f'Prediction: {prediction}')

    # evaluate
    error = loss(prediction, label)
    print(f'Mean Squared Error: {error: .4f}')

    # train
    delta = gradient(prediction, label)
    weight -= backward(example, delta)

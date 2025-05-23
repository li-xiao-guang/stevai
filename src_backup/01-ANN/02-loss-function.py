import numpy as np


# neuron definition
def predict(x, w):
    return x.dot(w.T)


# loss function (mean squared error)
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# input
example = np.array([25.5, 65.0, 800])
label = np.array([2.0, 8.0])

# layer definition (out_size, in_size)
weight = np.array([[0.5, 0.5, 0.5],
                   [1.0, 1.0, 1.0]])

# output
prediction = predict(example, weight)
print(f'Prediction: {prediction}')

# evaluate
error = mse_loss(prediction, label)
print(f'Mean Squared Error:{error: .4f}')

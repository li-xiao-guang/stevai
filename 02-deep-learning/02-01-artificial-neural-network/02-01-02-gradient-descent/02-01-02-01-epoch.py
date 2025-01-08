import numpy as np


# neuron definition
def forward(x, w):
    return np.dot(x, w)


# loss function
def loss(p, y):
    return ((p - y) ** 2).mean()


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
    print("Mean Squared Error: ", f'{error: .4f}')

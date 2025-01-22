import numpy as np


# neuron definition
def predict(x, w):
    return x.dot(w.T)


# layer definition (out_size, in_size)
weight = np.array([[0.5, 0.5, 0.5],
                   [1.0, 1.0, 1.0]])

# input
example = np.array([25.5, 65.0, 800])

# output
prediction = predict(example, weight)
print("Prediction: ", prediction)

# neuron definition
def forward(x, w):
    return x * w


# layer connection definition
weight = 0.5

# input
example = 2.0

# output
prediction = forward(example, weight)
print("Prediction: ", prediction)

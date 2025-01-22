import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

    def dot(self, other: 'Tensor'):
        return Tensor(self.data.dot(other.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    np.random.seed(0)

    examples = Tensor(np.array([[25.5, 65.0, 800],
                                [18.2, 45.0, 400],
                                [32.1, 75.0, 900],
                                [22.3, 62.0, 750],
                                [35.0, 80.0, 950],
                                [20.1, 55.0, 600],
                                [28.4, 70.0, 850]]))
    labels = Tensor(np.array([[0.9, 0.4],
                              [0.3, 0.2],
                              [0.4, 0.5],
                              [0.8, 0.3],
                              [0.2, 0.5],
                              [0.6, 0.3],
                              [0.7, 0.4]]))

    weights = list()
    weights.append(Tensor(np.random.rand(3, 8)))
    weights.append(Tensor(np.random.rand(8, 2)))

    predictions = examples.dot(weights[0]).dot(weights[1])
    print(predictions)

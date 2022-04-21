import numpy as np
import numpy.typing as npt

from .layer import Layer

class Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: npt.ArrayLike) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

    
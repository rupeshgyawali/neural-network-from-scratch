import numpy as np
import numpy.typing as npt

from .activation import Activation

class Softmax(Activation):
    def forward(self, inputs: npt. ArrayLike) -> None:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


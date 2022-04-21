import numpy as np
import numpy.typing as npt

from .activation import Activation

class Softmax(Activation):
    def forward(self, inputs: npt. ArrayLike) -> None:
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: npt.ArrayLike) -> None:
        self.inputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


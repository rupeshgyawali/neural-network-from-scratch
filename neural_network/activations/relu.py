import numpy as np
import numpy.typing as npt

from .activation import Activation

class ReLU(Activation):
    def forward(self, inputs: npt.ArrayLike) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: npt.ArrayLike) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


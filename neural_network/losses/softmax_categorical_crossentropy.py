import numpy as np
import numpy.typing as npt

from ..activations import Softmax, Activation
from .categorical_crossentropy import CategoricalCrossentropy
from .loss import Loss

class Softmax_CategoricalCrossentropy(Activation, Loss):
    """Combination of Softmax activation and Categorical Crossentropy
    inorder speed up gradient calculation.
    """
    def __init__(self) -> None:
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()
    
    def forward(self, inputs: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues: npt.ArrayLike, y_true: npt.ArrayLike) -> None:
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


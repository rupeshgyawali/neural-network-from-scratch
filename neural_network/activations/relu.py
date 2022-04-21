import numpy as np
import numpy.typing as npt

from .activation import Activation

class ReLU(Activation):
    def forward(self, inputs: npt.ArrayLike) -> None:
        self.output = np.maximum(0, inputs)


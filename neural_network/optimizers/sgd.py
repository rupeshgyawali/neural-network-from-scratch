from typing import Type

from ..layers import Layer
from .optimizer import Optimizer

class SGD(Optimizer):
    def update_params(self, layer: Type[Layer]) -> None:
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
from typing import Type

from ..layers import Layer

class Optimizer:
    def __init__(self, learning_rate: float=1.0) -> None:
        self.learning_rate = learning_rate

    def update_params(self, layer: Type[Layer]) -> None:
        raise NotImplemented
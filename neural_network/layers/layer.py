import numpy.typing as npt

class Layer:
    def __init__(self) -> None:
        self.output = None

    def forward(self, inputs: npt.ArrayLike) -> None:
        raise NotImplemented

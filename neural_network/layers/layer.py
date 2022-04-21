import numpy.typing as npt

class Layer:
    def __init__(self) -> None:
        self.inputs = None
        self.output = None

    def forward(self, inputs: npt.ArrayLike) -> None:
        raise NotImplemented

    def backward(self, dvalues: npt.ArrayLike) -> None:
        raise NotImplemented


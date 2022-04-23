import numpy as np
import numpy.typing as npt
from typing import List, Type

from .layers import Layer
from .activations import Activation
from .losses import Loss
from .optimizers import Optimizer

class Model:
    def __init__(self, layers: List[Type[Layer]], 
                       optimizer: Type[Optimizer],
                       loss: Type[Loss]) -> None:
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def train(self, X: npt.ArrayLike, y: npt.ArrayLike, epochs: int) -> None:
        for epoch in range(1, epochs+1):
            self._forward(X)
            
            # loss calculation
            loss = self.loss.forward(self.layers[-1].output, y)
            
            # Accuracy calculation
            predictions = np.argmax(self.loss.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions==y)
            if not epoch % 100:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')
            
            self._backward(y)
            self._update()

    def _forward(self, X: npt.ArrayLike) -> None:
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(X)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def _backward(self, y: npt.ArrayLike) -> None:
        self.loss.backward(self.loss.output, y)
        layers_len = len(self.layers)
        for i in range(layers_len-1, -1, -1):
            if i == layers_len - 1:
                self.layers[i].backward(self.loss.dinputs)
            else:
                self.layers[i].backward(self.layers[i+1].dinputs)
    
    def _update(self) -> None:
        for layer in self.layers:
            if not isinstance(layer, Activation):
                self.optimizer.update_params(layer)



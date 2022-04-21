import numpy as np
import numpy.typing as npt
from typing import Tuple

from neural_network.layers import Dense

np.random.seed(0)

# Create spiral dataset
def create_data(samples: int, classes: int) -> Tuple[npt.NDArray, npt.NDArray]:
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

layer1 = Dense(2, 3)
layer2 = Dense(3, 3)

# Forward propagation
layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output[:5])
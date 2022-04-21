import numpy as np
import numpy.typing as npt

class Loss:
    def calculate(self, output: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
        sample_losses = self.forward(output, y)
        batch_mean_loss = np.mean(sample_losses)
        return batch_mean_loss

    def forward(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplemented
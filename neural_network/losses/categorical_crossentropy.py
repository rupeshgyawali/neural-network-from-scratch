import numpy as np
import numpy.typing as npt

from .loss import Loss

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        samples = len(y_pred)
        # To prevent division by log of 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # For categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
            ]
        # For one-hot encoded labels
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log = -np.log(correct_confidences)
        return negative_log
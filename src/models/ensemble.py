"""Ensemble and stacking methods for out-of-fold probability blending."""

from typing import List
import numpy as np


class BlendedEnsemble:
    """Weighted soft-voting blender over out-of-fold probability distributions."""

    def __init__(self, weights: List[float]):
        self.weights = np.array(weights) / sum(weights)

    def blend(self, prob_matrices: List[np.ndarray]) -> np.ndarray:
        """Combines multiple predicted probability arrays using normalized weights."""
        weighted_sum = np.zeros_like(prob_matrices[0])
        for w, probs in zip(self.weights, prob_matrices):
            weighted_sum += w * probs
        return weighted_sum

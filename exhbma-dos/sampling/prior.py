from abc import ABC, abstractmethod

import numpy as np


class BasePrior(ABC):
    @abstractmethod
    def prob(self, x: float) -> float:
        raise NotImplementedError


class InversePrior(BasePrior):
    def __init__(self, low: float = 1e-3, high: float = 1e1):
        self.low = low
        self.high = high
        self.norm = np.log(high) - np.log(low)

    def prob(self, x: float) -> float:
        assert x >= 0, "x must be greater than or equal to zero."
        if x < self.low or x > self.high:
            return 0
        return (1 / x) / self.norm

from abc import ABC, abstractmethod

from routing.utils.vectorized2d import Coordinate


class Sampler(ABC):

    @abstractmethod
    def sample(self, num_of_samples: int = 1) -> Coordinate:
        pass

    @abstractmethod
    def register(self, x_new: Coordinate) -> None:
        pass

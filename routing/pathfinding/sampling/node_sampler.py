from routing.pathfinding.sampling.sampler import Sampler
from vectorized2d import Coordinate


class NodeSampler(Sampler):
    """
    Used to "sample" one specific node as part of the architecture.
    """
    def __init__(self, node: Coordinate):
        self._node = node

    def sample(self, num_of_samples: int = 1) -> Coordinate:
        return self._node

    def register(self, x_new: Coordinate) -> None:
        pass

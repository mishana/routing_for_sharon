from typing import Dict

from routing.setting.environment import Environment
from routing.pathfinding.extending.tree_extender import TreeExtender
from routing.pathfinding.sampling.node_sampler import NodeSampler
from routing.pathfinding.sampling.sampler import Sampler
from routing.vehicle.vehicle import Vehicle
from vectorized2d import Coordinate


class NodeTreeExtender(TreeExtender):
    def __init__(self, environment: Environment, vehicle: Vehicle, costs: Dict[Coordinate, float],
                 x: Coordinate) -> None:
        super().__init__(environment, vehicle, costs)
        self.__sampler: Sampler = NodeSampler(x)

    @property
    def _sampler(self) -> Sampler:
        return self.__sampler

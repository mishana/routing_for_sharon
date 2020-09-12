from typing import Dict

from vectorized2d import Coordinate

from routing.pathfinding.extending.tree_extender import TreeExtender
from routing.pathfinding.sampling.grid_sampler import GridSampler
from routing.pathfinding.sampling.sampler import Sampler
from routing.setting.environment import Environment
from routing.vehicle.vehicle import Vehicle


class RRTStarExtender(TreeExtender):
    def __init__(self, environment: Environment, vehicle: Vehicle, costs: Dict[Coordinate, float],
                 x_init: Coordinate, x_goal: Coordinate) -> None:
        super().__init__(environment, vehicle, costs)
        self.__sampler = GridSampler(x_init, x_goal)

    @property
    def _sampler(self) -> Sampler:
        return self.__sampler

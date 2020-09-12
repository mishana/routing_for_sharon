from abc import ABC, abstractmethod

from vectorized2d import Coordinate

from routing.setting.environment import Environment
from routing.vehicle.vehicle import Vehicle


class Planner(ABC):
    def __init__(self,
                 environment: Environment,
                 vehicle: Vehicle):
        self.env = environment
        self.vehicle = vehicle

    @abstractmethod
    def plan(self, x_init: Coordinate, x_goal: Coordinate, iterations_num: int) -> Coordinate:
        """

        :param x_init: the initial (source) coordinate
        :param x_goal: the goal (destination) coordinate
        :param iterations_num: number of iterations to run the planning
        :return: a vectorized Coordinate object containing the coordinates
                 on the shortest path from x_init to x_goal (in order)
        """
        pass

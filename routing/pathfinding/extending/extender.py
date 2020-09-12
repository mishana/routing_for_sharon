from abc import ABC, abstractmethod

from routing.pathfinding.graphs.planning_graph import PlanningGraph
from routing.setting.environment import Environment
from routing.vehicle.vehicle import Vehicle


class Extender(ABC):
    """ Performs pre-defined logic to extend the planning graph """

    def __init__(self, environment: Environment, vehicle: Vehicle) -> None:
        """ Initialize a new Extender object

        :param environment: data object describing the environment
        :param vehicle: data object describing the vehicle
        """
        self.env = environment
        self.vehicle = vehicle

    @abstractmethod
    def extend(self, graph: PlanningGraph) -> bool:
        """ Performing one iteration to extend the current graph

        :param graph: the current planning graph
        :return: True if the graph had been modified else False
        """
        pass

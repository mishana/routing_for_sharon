from abc import ABC, abstractmethod
from typing import Dict, Tuple, Iterable

import numpy as np
from vectorized2d import Coordinate

from routing.pathfinding.extending.extender import Extender
from routing.pathfinding.graphs.planning_tree import PlanningTree
from routing.pathfinding.sampling.sampler import Sampler
from routing.setting.environment import Environment
from routing.vehicle.vehicle import Vehicle


class TreeExtender(Extender, ABC):
    def __init__(self, environment: Environment, vehicle: Vehicle, costs: Dict[Coordinate, float]) -> None:
        super().__init__(environment, vehicle)
        self._costs: Dict[Coordinate, float] = costs

    @property
    @abstractmethod
    def _sampler(self) -> Sampler:
        """ Defines the sampler the extender will use to choose the next node"""
        pass

    @staticmethod
    def _steer_and_find_near(graph: PlanningTree, x: Coordinate) -> Tuple[Coordinate, Coordinate, Iterable[int]]:
        """

        :param graph: The graph to operate on.
        :param x: new sampled Coordinate(s).
        :return: a tuple with three elements:
                    1. a steered x.
                    2. a multi-coordinate which holds all the nodes coordinates that are "near" x, for every
                        single-coordinate in x (vertically stacked).
                    3. a 1D numpy array of indices, stating what coordinate in x corresponds to the "near" coordinates.

        """
        x_nearest = graph.nearest(x)
        x_new = graph.steer(x_from=x_nearest, x_towards=x)
        x_near, x_indices = graph.near(x=x_new, n=graph.number_of_nodes() - 1)

        x_indices = x_indices.tolist()
        no_near_indices = list(set(range(len(x))) - set(x_indices))
        return (x_new,
                Coordinate.concat([x_near, x_nearest[no_near_indices]]),
                x_indices + no_near_indices)

    def extend(self, graph: PlanningTree) -> None:
        # Sample new node
        x = self._sampler.sample()

        # Find all safe edges and choose the best one
        x_new, x_connect_candidates, x_connect_indices = self._steer_and_find_near(graph, x)
        mask = self.env.is_obstacle_free(x_from=x_connect_candidates, x_to=x_new, vehicle=self.vehicle)
        if not self._connect_new(graph=graph, x_source=x_connect_candidates, x_target=x_new, mask=mask):
            return

        # The node has been connected, register it
        self._sampler.register(x)

        # Find safe edges for rewiring and choose the ones the shorten the path
        mask = self.env.is_obstacle_free(x_from=x_new, x_to=x_connect_candidates, vehicle=self.vehicle)
        costs_source_to_target = x_new.geo_dist(x_connect_candidates)
        self._attempt_rewire(graph, x_new, x_connect_candidates[mask], costs_source_to_target[mask])

    def _connect_new(self, graph: PlanningTree, x_source: Coordinate, x_target: Coordinate, mask: np.ndarray) -> bool:
        if np.count_nonzero(mask) == 0:
            return False
        # First find the best source through which to connect the target
        costs_to_source = np.array([self._costs[candidate] for candidate in x_source])
        costs_source_to_target = x_source.geo_dist(x_target)
        costs_to_target = (costs_to_source + costs_source_to_target)[mask]
        idx_min = np.argmin(costs_to_target)

        # Then add the target with the appropriate weight
        graph.add_edge(x_source[mask][idx_min], x_target, dist=costs_source_to_target[mask][idx_min])
        self._costs[x_target] = costs_to_target[idx_min]
        return True

    def _attempt_rewire(self, graph: PlanningTree, x_new: Coordinate, rewire_candidates: Coordinate,
                        rewiring_costs: np.ndarray) -> None:
        path_costs = self._costs[x_new] + rewiring_costs
        for i, rewire_candidate in enumerate(rewire_candidates):
            if self._costs[rewire_candidate] > path_costs[i]:
                graph.remove_edge(graph.get_parent(rewire_candidate), rewire_candidate)
                graph.add_edge(x_new, rewire_candidate, dist=rewiring_costs[i])
                self._costs[rewire_candidate] = path_costs[i]

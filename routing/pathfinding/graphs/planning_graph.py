import math
from typing import Tuple

import networkx as nx
import numpy as np
from vectorized2d import Coordinate


class PlanningGraph(nx.DiGraph):
    def __init__(self, steering_coeff: float, gamma: float, **attr):
        super().__init__(**attr)
        self.steering_coeff = steering_coeff
        self.gamma = gamma

    def nearest(self, x: Coordinate) -> Coordinate:
        """
        Finds a node(s) in the graph that is the "closest" to x, in terms of euclidean distance.

        :param x: the reference coordinate(s)
        :return: the nearest neighbor(s) of x in the graph
        """
        coordinates = Coordinate.concat(list(self))
        # used instead of geo-distance for better performance time-wise
        dists = x.euclid_dist_squared(coordinates)  # shape = (len(x), len(coordinates))
        return coordinates[np.argmin(dists, axis=1)]

    def steer(self, x_from: Coordinate, x_towards: Coordinate) -> Coordinate:
        """
        Calculate a new point(s), that is the result of steering x_from towards x_towards,
        within a steering coefficient.

        Note: x_from and x_towrads must be of same shape.

        :param x_from: the source coordinate(s) of the steering operation
        :param x_towards: the destination coordinate(s) of the steering operation
        :return: a coordinate(s) on the line between x_from and x_towards, within a steering coefficient
        """
        # used instead of geo-distance for better performance time-wise
        is_far = x_from.euclid_dist_squared(x_towards, pairing=Coordinate.Pairing.ALIGNED) >= self.steering_coeff ** 2
        x_from_far, x_towards_far = x_from[is_far], x_towards[is_far]

        steered = x_towards.copy()
        steered[is_far] = x_from_far + (x_towards_far - x_from_far).normalized() * self.steering_coeff

        return steered

    def near(self, x: Coordinate, n: int) -> Tuple[Coordinate, np.ndarray]:
        """
        Finds a collection of nodes in the graph that reside within a (calculated, a parameter of n) radius from x,
        in terms of euclidean distance.

        Examples:
        --------
        >>> g = PlanningGraph(0., 0.)
        >>> g.add_nodes_from(Coordinate(lat=[1., 2.], lon=[3., 4.]))
        >>> list(g)
        [Coordinate([[1., 3.]]), Coordinate([[2., 4.]])]
        >>> x = Coordinate(lat=[1., 2.], lon=[3., 4.])

        >>> near_x, x_indices = g.near(x=x, n=g.number_of_nodes() - 1)
        >>> near_x
        Coordinate([[1., 3.],
                    [2., 4.]])
        >>> x_indices
        array([0, 1])
        >>> near_0 = near_x[x_indices == 0]
        >>> near_1 = near_x[x_indices == 1]
        >>> near_0
        Coordinate([[1., 3.]])
        >>> near_1
        Coordinate([[2., 4.]])

        :param x: the reference coordinate(s)
        :param n: a parameter for radius calculation (usually, it is the number of nodes in the graph)
        :return: a tuple with two elements:
                    1. multi-coordinate which holds all the nodes coordinates that are "near" x, for every
                        single-coordinate in x (vertically stacked).
                    2. a 1D numpy array of indices, stating what coordinate in x corresponds to the "near" coordinates.
        """
        r = min(self.steering_coeff, ((self.gamma / math.pi) * (math.log(n + 2) / (n + 2))) ** (1 / 2))

        coordinates = Coordinate.concat(list(self))
        # used instead of geo-distance for better performance time-wise
        dists = x.euclid_dist_squared(coordinates)  # shape = (len(x), len(coordinates))
        x_indices, c_indices = (dists <= r ** 2).nonzero()
        return coordinates[c_indices], x_indices

from typing import Dict, Optional

import networkx as nx
from vectorized2d import Coordinate

from routing.pathfinding.extending.node_tree_extender import NodeTreeExtender
from routing.pathfinding.extending.rrt_star_extender import RRTStarExtender
from routing.pathfinding.graphs.planning_tree import PlanningTree
from routing.pathfinding.planning.planner import Planner
from routing.setting.environment import Environment
from routing.vehicle.vehicle import Vehicle

TARGET_CONNECT_ATTEMPT_INTERVAL = 30  # tunable hyper parameter


class RRTStarPlanner(Planner):
    def __init__(self,
                 environment: Environment,
                 vehicle: Vehicle,
                 steering_coeff: float,  # see paper regarding RRT*
                 gamma: float  # see paper regarding RRT*
                 ):
        super().__init__(environment, vehicle)
        self.steering_coeff = steering_coeff
        self.gamma = gamma
        self.graph = PlanningTree(steering_coeff=self.steering_coeff, gamma=self.gamma)
        self._costs: Dict[Coordinate, float] = {}

    def plan(self, x_init: Coordinate, x_goal: Coordinate, iterations_num: int) -> Optional[Coordinate]:
        self.graph.add_node(x_init)
        self._costs[x_init] = 0
        extender = RRTStarExtender(self.env, self.vehicle, self._costs, x_init, x_goal)
        goal_extender = NodeTreeExtender(self.env, self.vehicle, self._costs, x_goal)

        for i in range(iterations_num):
            if i % TARGET_CONNECT_ATTEMPT_INTERVAL != 0:
                extender.extend(self.graph)
            else:
                # Try to add target to the tree, recalculate the cost dictionary before in order to make
                # sure that target will connect to the best path
                self._costs = nx.single_source_dijkstra_path_length(self.graph, x_init)
                if x_goal not in self.graph.nodes:
                    goal_extender.extend(self.graph)

        if x_init in self.graph and x_goal in self.graph:
            return Coordinate.concat(nx.dijkstra_path(self.graph, x_init, x_goal))

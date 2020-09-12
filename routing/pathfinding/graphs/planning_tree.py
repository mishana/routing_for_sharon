from vectorized2d import Coordinate

from routing.pathfinding.graphs.planning_graph import PlanningGraph


class PlanningTree(PlanningGraph):
    def get_parent(self, x: Coordinate) -> Coordinate:
        """
        Returns the parent node of x, under the assumption that the graph is a tree.

        :param x: the coordinates of a node in the graph
        :return: the parent of x in the tree (if exists), else None
        """
        predecessors = list(self.pred[x])
        return predecessors[0] if predecessors else None

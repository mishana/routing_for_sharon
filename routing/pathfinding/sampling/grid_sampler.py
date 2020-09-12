import numpy as np
from numba import njit
from vectorized2d import Coordinate

from routing.pathfinding.sampling.sampler import Sampler

BOUNDARY_BUFFER = 0.01


class GridSampler(Sampler):

    def __init__(self, source: Coordinate, target: Coordinate, size: int = 50):
        self.grid_size = size
        self.south = min(source.lat, target.lat) - BOUNDARY_BUFFER
        self.north = max(source.lat, target.lat) + BOUNDARY_BUFFER
        self.west = min(source.lon, target.lon) - BOUNDARY_BUFFER
        self.east = max(source.lon, target.lon) + BOUNDARY_BUFFER
        self.weights_grid = np.ones((self.grid_size, self.grid_size))
        self.rng = np.random.default_rng()

    @staticmethod
    @njit
    def update_weights(weights_grid, rows, cols):
        for r, c in zip(rows, cols):
            weights_grid[r, c] += 1

    def sample(self, num_of_samples: int = 1) -> Coordinate:
        current_inverse_grid = 1 / self.weights_grid
        sampled_idxs = self.rng.choice(np.arange(self.grid_size ** 2), num_of_samples,
                                       p=current_inverse_grid.ravel() / np.sum(current_inverse_grid))
        rows, cols = self._location_to_grid_index(sampled_idxs)

        sampled_tile_locations = Coordinate(lat=self.south + (self.north - self.south) * rows / self.grid_size,
                                            lon=self.west + (self.east - self.west) * cols / self.grid_size)
        intile_locations = Coordinate(
            lat=self.rng.uniform(low=0, high=(self.north - self.south) / self.grid_size, size=num_of_samples),
            lon=self.rng.uniform(low=0, high=(self.east - self.west) / self.grid_size, size=num_of_samples))

        self.update_weights(self.weights_grid, rows, cols)
        return sampled_tile_locations + intile_locations

    def register(self, x_new: Coordinate) -> None:
        rows = ((x_new.lat - self.south) / self.grid_size).astype(int)
        cols = ((x_new.lon - self.west) / self.grid_size).astype(int)
        self.update_weights(self.weights_grid, rows, cols)

    def _grid_index_to_location(self, row, col):
        return row * self.grid_size + col

    def _location_to_grid_index(self, location):
        return (location / self.grid_size).astype(int), location % self.grid_size

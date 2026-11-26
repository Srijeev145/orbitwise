"""Study region and analysis grid. Default: Karnataka bounding box (state boundary mask optional)."""
import numpy as np

REGION = {"name": "Karnataka", "south": 11.5, "north": 18.5, "west": 74.0, "east": 78.6}
CELL = 0.1  # degrees (~11 km); each cell is one candidate site


def grid_shape():
    return (round((REGION["north"] - REGION["south"]) / CELL),
            round((REGION["east"] - REGION["west"]) / CELL))


def cell_centers():
    """2-D arrays (rows north->south, cols west->east) of cell-center lat / lon."""
    nr, nc = grid_shape()
    lats = REGION["north"] - CELL * (np.arange(nr) + 0.5)
    lons = REGION["west"] + CELL * (np.arange(nc) + 0.5)
    return np.meshgrid(lats, lons, indexing="ij")

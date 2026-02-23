"""Grid-based terrain map with procedural generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

from village_sim.core.config import (
    MAP_HEIGHT,
    MAP_WIDTH,
    TERRAIN_COSTS,
    VILLAGE_CENTER,
)


@dataclass
class TerrainCell:
    """A single cell in the world grid."""

    terrain_type: str = "grassland"
    elevation: float = 0.0
    has_road: bool = False
    has_bridge: bool = False
    resource_node_id: Optional[int] = None
    structure_ids: list[int] = field(default_factory=list)


class WorldMap:
    """Grid-based world map with procedural terrain generation."""

    def __init__(self, width: int = MAP_WIDTH, height: int = MAP_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.grid: list[list[TerrainCell]] = [
            [TerrainCell() for _ in range(width)] for _ in range(height)
        ]

    def generate(self, rng: Generator) -> None:
        """Procedurally generate terrain using value noise."""
        elevation = _generate_noise(self.width, self.height, rng, scale=20.0, octaves=4)
        moisture = _generate_noise(self.width, self.height, rng, scale=25.0, octaves=3)

        for y in range(self.height):
            for x in range(self.width):
                e = elevation[y][x]
                m = moisture[y][x]
                self.grid[y][x].elevation = e
                self.grid[y][x].terrain_type = _classify_terrain(e, m)

        # Clear village center area
        cx, cy = VILLAGE_CENTER
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.grid[ny][nx].terrain_type = "grassland"
                    self.grid[ny][nx].elevation = 0.3

        # Ensure at least one river
        _carve_river(self, rng)

    def get_cell(self, x: int, y: int) -> TerrainCell:
        """Get the terrain cell at (x, y)."""
        return self.grid[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def movement_cost(self, x: int, y: int) -> float:
        """Get the movement cost for entering cell (x, y)."""
        cell = self.grid[y][x]
        if cell.has_road:
            return TERRAIN_COSTS["path"]
        if cell.terrain_type == "river" and cell.has_bridge:
            return TERRAIN_COSTS["path"]
        return TERRAIN_COSTS.get(cell.terrain_type, 2.0)

    def cells_in_radius(
        self, center: tuple[int, int], radius: int
    ) -> list[tuple[int, int]]:
        """Return all (x, y) positions within Manhattan radius of center."""
        cx, cy = center
        result: list[tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if self.in_bounds(nx, ny):
                    dist = abs(dx) + abs(dy)
                    if dist <= radius:
                        result.append((nx, ny))
        return result

    def neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return walkable 4-directional neighbors."""
        result: list[tuple[int, int]] = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                cell = self.grid[ny][nx]
                # River impassable without bridge
                if cell.terrain_type == "river" and not cell.has_bridge:
                    continue
                result.append((nx, ny))
        return result


# ---------------------------------------------------------------------------
# Terrain generation helpers
# ---------------------------------------------------------------------------

def _generate_noise(
    width: int, height: int, rng: Generator, scale: float = 20.0, octaves: int = 4
) -> list[list[float]]:
    """Generate 2D value noise in [0, 1] using numpy interpolation."""
    result = np.zeros((height, width), dtype=np.float64)
    for octave in range(octaves):
        freq = 2 ** octave
        amp = 0.5 ** octave
        # Generate random grid at this frequency
        gw = max(2, int(width / scale * freq) + 2)
        gh = max(2, int(height / scale * freq) + 2)
        grid = rng.random((gh, gw))

        # Interpolate to full resolution
        xs = np.linspace(0, gw - 1, width)
        ys = np.linspace(0, gh - 1, height)
        # Bilinear interpolation
        xi = np.clip(xs.astype(int), 0, gw - 2)
        yi = np.clip(ys.astype(int), 0, gh - 2)
        xf = xs - xi
        yf = ys - yi

        for y_idx in range(height):
            y0 = yi[y_idx]
            fy = yf[y_idx]
            for x_idx in range(width):
                x0 = xi[x_idx]
                fx = xf[x_idx]
                v00 = grid[y0, x0]
                v10 = grid[y0, x0 + 1]
                v01 = grid[y0 + 1, x0]
                v11 = grid[y0 + 1, x0 + 1]
                v = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + \
                    v01 * (1 - fx) * fy + v11 * fx * fy
                result[y_idx, x_idx] += v * amp

    # Normalize to [0, 1]
    rmin, rmax = result.min(), result.max()
    if rmax > rmin:
        result = (result - rmin) / (rmax - rmin)
    return result.tolist()


def _classify_terrain(elevation: float, moisture: float) -> str:
    """Map elevation and moisture to a terrain type."""
    if elevation > 0.85:
        return "mountain"
    if elevation > 0.7:
        return "rocky"
    if elevation > 0.6:
        return "hills"
    if elevation < 0.15 and moisture > 0.6:
        return "swamp"
    if moisture > 0.7:
        return "dense_forest"
    if moisture > 0.5:
        return "light_forest"
    return "grassland"


def _carve_river(world_map: WorldMap, rng: Generator) -> None:
    """Carve a river across the map from a high to low point."""
    # Start near the top, meander to the bottom
    x = rng.integers(world_map.width // 4, 3 * world_map.width // 4)
    for y in range(world_map.height):
        world_map.grid[y][x].terrain_type = "river"
        world_map.grid[y][x].elevation = 0.05
        # Meander
        drift = rng.integers(-1, 2)  # -1, 0, or 1
        x = max(1, min(world_map.width - 2, x + drift))

    # Place a bridge near village center
    cx, cy = VILLAGE_CENTER
    best_dist = float("inf")
    bridge_pos = None
    for y in range(world_map.height):
        for bx in range(world_map.width):
            if world_map.grid[y][bx].terrain_type == "river":
                d = abs(bx - cx) + abs(y - cy)
                if d < best_dist:
                    best_dist = d
                    bridge_pos = (bx, y)
    if bridge_pos:
        bx, by = bridge_pos
        world_map.grid[by][bx].has_bridge = True

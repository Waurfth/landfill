"""Resource nodes placed on the world map."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from numpy.random import Generator

from village_sim.core.config import (
    FISH_REGEN_RATE,
    FOREST_REGEN_RATE,
    HERB_REGEN_RATE,
    MINE_REGEN_RATE,
    WILD_PLANTS_REGEN_RATE,
)

# Avoid circular import — WorldMap is passed as argument, not imported at module level.


class ResourceType(Enum):
    TIMBER = "timber"
    GAME_SMALL = "game_small"
    GAME_LARGE = "game_large"
    FISH = "fish"
    STONE = "stone"
    CLAY = "clay"
    IRON_ORE = "iron_ore"
    WILD_PLANTS = "wild_plants"
    MEDICINAL_HERBS = "medicinal_herbs"
    FARMLAND = "farmland"
    FRESH_WATER = "fresh_water"


# Default regeneration rates per resource type
_REGEN_RATES: dict[ResourceType, float] = {
    ResourceType.TIMBER: FOREST_REGEN_RATE,
    ResourceType.GAME_SMALL: 0.015,
    ResourceType.GAME_LARGE: 0.008,
    ResourceType.FISH: FISH_REGEN_RATE,
    ResourceType.STONE: MINE_REGEN_RATE,
    ResourceType.CLAY: 0.001,
    ResourceType.IRON_ORE: MINE_REGEN_RATE,
    ResourceType.WILD_PLANTS: WILD_PLANTS_REGEN_RATE,
    ResourceType.MEDICINAL_HERBS: HERB_REGEN_RATE,
    ResourceType.FARMLAND: 0.0,
    ResourceType.FRESH_WATER: 1.0,  # always full
}

# Seasonal modifiers: resource_type -> {season: multiplier}
_SEASONAL_MODIFIERS: dict[ResourceType, dict[str, float]] = {
    ResourceType.FISH: {"spring": 1.3, "summer": 1.0, "autumn": 0.8, "winter": 0.5},
    ResourceType.GAME_SMALL: {"spring": 1.2, "summer": 1.0, "autumn": 0.9, "winter": 0.6},
    ResourceType.GAME_LARGE: {"spring": 1.0, "summer": 0.9, "autumn": 1.3, "winter": 0.7},
    ResourceType.WILD_PLANTS: {"spring": 1.3, "summer": 1.5, "autumn": 0.8, "winter": 0.2},
    ResourceType.MEDICINAL_HERBS: {"spring": 1.2, "summer": 1.4, "autumn": 0.6, "winter": 0.1},
}


@dataclass
class ResourceNode:
    """A harvestable resource at a map position."""

    node_id: int
    resource_type: ResourceType
    position: tuple[int, int]
    max_abundance: float
    current_abundance: float
    regeneration_rate: float
    seasonal_modifier: dict[str, float] = field(default_factory=lambda: {
        "spring": 1.0, "summer": 1.0, "autumn": 1.0, "winter": 1.0,
    })
    danger_level: float = 0.0
    required_tools: list[str] = field(default_factory=list)

    def harvest(self, amount: float, tool_quality: float = 1.0) -> float:
        """Harvest up to *amount* from this node. Returns actual yield."""
        effective = min(amount * tool_quality, self.current_abundance)
        self.current_abundance -= effective
        return effective

    def regenerate(self, season: str) -> None:
        """Regenerate toward max_abundance, modified by season."""
        modifier = self.seasonal_modifier.get(season, 1.0)
        growth = self.regeneration_rate * self.max_abundance * modifier
        self.current_abundance = min(self.max_abundance, self.current_abundance + growth)


class ResourceManager:
    """Manages all resource nodes in the world."""

    def __init__(self) -> None:
        self._nodes: dict[int, ResourceNode] = {}
        self._next_id: int = 0

    @property
    def nodes(self) -> list[ResourceNode]:
        return list(self._nodes.values())

    def get_node(self, node_id: int) -> Optional[ResourceNode]:
        return self._nodes.get(node_id)

    def add_node(self, node: ResourceNode) -> None:
        self._nodes[node.node_id] = node

    def generate_resources(self, world_map: "WorldMap", rng: Generator) -> None:  # noqa: F821
        """Place resource nodes on appropriate terrain."""
        from village_sim.world.map import WorldMap  # local import to avoid circular

        for y in range(world_map.height):
            for x in range(world_map.width):
                cell = world_map.get_cell(x, y)
                node = self._maybe_create_node(cell.terrain_type, x, y, rng)
                if node is not None:
                    self.add_node(node)
                    cell.resource_node_id = node.node_id

    def daily_regeneration(self, season: str) -> None:
        """Regenerate all resource nodes."""
        for node in self._nodes.values():
            node.regenerate(season)

    def get_nearest_of_type(
        self, position: tuple[int, int], resource_type: ResourceType
    ) -> Optional[ResourceNode]:
        """Find the nearest node of a given type."""
        best: Optional[ResourceNode] = None
        best_dist = float("inf")
        px, py = position
        for node in self._nodes.values():
            if node.resource_type == resource_type and node.current_abundance > 0:
                nx, ny = node.position
                dist = abs(nx - px) + abs(ny - py)
                if dist < best_dist:
                    best_dist = dist
                    best = node
        return best

    def get_all_in_radius(
        self,
        position: tuple[int, int],
        radius: int,
        resource_type: Optional[ResourceType] = None,
    ) -> list[ResourceNode]:
        """Get all nodes within Manhattan radius, optionally filtered by type."""
        px, py = position
        result: list[ResourceNode] = []
        for node in self._nodes.values():
            if resource_type is not None and node.resource_type != resource_type:
                continue
            nx, ny = node.position
            if abs(nx - px) + abs(ny - py) <= radius:
                result.append(node)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_node_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _maybe_create_node(
        self, terrain: str, x: int, y: int, rng: Generator
    ) -> Optional[ResourceNode]:
        """Probabilistically create a resource node based on terrain type."""
        # Mapping: terrain -> possible (resource_type, probability, max_abundance, danger, tools)
        options: list[tuple[ResourceType, float, float, float, list[str]]] = []

        if terrain == "light_forest":
            options.append((ResourceType.TIMBER, 0.3, 50.0, 0.02, ["axe"]))
            options.append((ResourceType.WILD_PLANTS, 0.2, 30.0, 0.01, []))
            options.append((ResourceType.GAME_SMALL, 0.15, 20.0, 0.03, []))
        elif terrain == "dense_forest":
            options.append((ResourceType.TIMBER, 0.5, 80.0, 0.04, ["axe"]))
            options.append((ResourceType.GAME_LARGE, 0.2, 15.0, 0.12, ["spear"]))
            options.append((ResourceType.GAME_SMALL, 0.25, 25.0, 0.05, []))
            options.append((ResourceType.MEDICINAL_HERBS, 0.1, 10.0, 0.02, []))
        elif terrain == "river":
            options.append((ResourceType.FISH, 0.6, 40.0, 0.02, ["fishing"]))
            options.append((ResourceType.FRESH_WATER, 0.9, 100.0, 0.0, []))
        elif terrain == "rocky":
            options.append((ResourceType.STONE, 0.4, 60.0, 0.06, ["pickaxe"]))
            options.append((ResourceType.IRON_ORE, 0.1, 20.0, 0.10, ["pickaxe"]))
        elif terrain == "hills":
            options.append((ResourceType.STONE, 0.2, 40.0, 0.05, ["pickaxe"]))
            options.append((ResourceType.CLAY, 0.15, 30.0, 0.01, []))
            options.append((ResourceType.GAME_SMALL, 0.1, 15.0, 0.04, []))
        elif terrain == "grassland":
            options.append((ResourceType.WILD_PLANTS, 0.1, 20.0, 0.01, []))
            options.append((ResourceType.FARMLAND, 0.08, 100.0, 0.0, []))
        elif terrain == "swamp":
            options.append((ResourceType.CLAY, 0.3, 40.0, 0.04, []))
            options.append((ResourceType.MEDICINAL_HERBS, 0.15, 15.0, 0.05, []))

        for rtype, prob, max_ab, danger, tools in options:
            if rng.random() < prob:
                seasonal = _SEASONAL_MODIFIERS.get(
                    rtype,
                    {"spring": 1.0, "summer": 1.0, "autumn": 1.0, "winter": 1.0},
                )
                node = ResourceNode(
                    node_id=self._next_node_id(),
                    resource_type=rtype,
                    position=(x, y),
                    max_abundance=max_ab,
                    current_abundance=max_ab * rng.uniform(0.5, 1.0),
                    regeneration_rate=_REGEN_RATES.get(rtype, 0.01),
                    seasonal_modifier=dict(seasonal),
                    danger_level=danger,
                    required_tools=list(tools),
                )
                return node
        return None

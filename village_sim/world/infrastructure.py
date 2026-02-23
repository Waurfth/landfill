"""Buildings and improvements placed on the world map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from village_sim.core.config import (
    SHELTER_CAPACITY_BASE,
    SHELTER_DAILY_DEGRADATION,
    STORM_DEGRADATION_MULTIPLIER,
)


@dataclass
class Structure:
    """A building or improvement on the map."""

    structure_id: int
    structure_type: str  # "shelter", "storage_shed", "road_segment", "bridge", "well", "farm_plot"
    position: tuple[int, int]
    quality: float = 0.5        # 0-1, affects effectiveness
    durability: float = 1.0     # 0-1, degrades over time
    owner_family_id: Optional[int] = None  # None = communal
    capacity: float = SHELTER_CAPACITY_BASE  # people for shelters, weight for storage


class InfrastructureManager:
    """Manages all structures in the world."""

    def __init__(self) -> None:
        self._structures: dict[int, Structure] = {}
        self._next_id: int = 0

    @property
    def structures(self) -> list[Structure]:
        return list(self._structures.values())

    def _next_structure_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def add_structure(self, structure: Structure) -> None:
        self._structures[structure.structure_id] = structure

    def create_structure(
        self,
        structure_type: str,
        position: tuple[int, int],
        quality: float = 0.5,
        owner_family_id: Optional[int] = None,
    ) -> Structure:
        """Create and register a new structure."""
        s = Structure(
            structure_id=self._next_structure_id(),
            structure_type=structure_type,
            position=position,
            quality=quality,
            owner_family_id=owner_family_id,
        )
        self.add_structure(s)
        return s

    def get_shelter_for(self, family_id: int) -> Optional[Structure]:
        """Get the shelter belonging to a family."""
        for s in self._structures.values():
            if s.structure_type == "shelter" and s.owner_family_id == family_id:
                return s
        return None

    def get_communal_structures(self) -> list[Structure]:
        return [s for s in self._structures.values() if s.owner_family_id is None]

    def daily_degradation(self, weather_damage_modifier: float = 1.0) -> None:
        """Degrade all structures slightly. Weather accelerates damage."""
        for s in self._structures.values():
            s.durability -= SHELTER_DAILY_DEGRADATION * weather_damage_modifier
            s.durability = max(0.0, s.durability)

    def repair(self, structure_id: int, repair_amount: float) -> None:
        """Improve durability of a structure."""
        s = self._structures.get(structure_id)
        if s:
            s.durability = min(1.0, s.durability + repair_amount)

    def shelter_quality_for(self, family_id: int) -> float:
        """Effective shelter quality for a family, accounting for durability."""
        shelter = self.get_shelter_for(family_id)
        if shelter is None:
            return 0.0
        return shelter.quality * shelter.durability

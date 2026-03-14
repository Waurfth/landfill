"""Buildings and improvements placed on the world map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from village_sim.core.config import (
    GRANARY_SAFETY_BONUS,
    MEETING_HALL_COMFORT_BONUS,
    MEETING_HALL_SOCIAL_BONUS,
    SHELTER_CAPACITY_BASE,
    SHELTER_DAILY_DEGRADATION,
    STORM_DEGRADATION_MULTIPLIER,
    WELL_THIRST_BONUS,
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

    def get_structure_of_type(self, structure_type: str) -> Optional[Structure]:
        """Get the first structure of a given type (for communal buildings)."""
        for s in self._structures.values():
            if s.structure_type == structure_type:
                return s
        return None

    def has_communal(self, structure_type: str) -> bool:
        """Check if a communal building of this type exists."""
        for s in self._structures.values():
            if s.structure_type == structure_type and s.owner_family_id is None:
                return True
        return False

    def get_village_bonuses(self) -> dict[str, float]:
        """Calculate aggregate need bonuses from all communal structures.

        Returns a dict of bonus_name -> bonus_value. Only includes bonuses
        for structures that actually exist and have durability > 0.
        """
        bonuses: dict[str, float] = {}

        for s in self._structures.values():
            if s.durability <= 0:
                continue
            effectiveness = s.quality * s.durability

            if s.structure_type == "well" and s.owner_family_id is None:
                bonuses["well"] = bonuses.get("well", 0) + WELL_THIRST_BONUS * effectiveness

            elif s.structure_type == "meeting_hall" and s.owner_family_id is None:
                bonuses["meeting_hall_social"] = (
                    bonuses.get("meeting_hall_social", 0) + MEETING_HALL_SOCIAL_BONUS * effectiveness
                )
                bonuses["meeting_hall_comfort"] = (
                    bonuses.get("meeting_hall_comfort", 0) + MEETING_HALL_COMFORT_BONUS * effectiveness
                )

            elif s.structure_type == "granary" and s.owner_family_id is None:
                bonuses["granary_safety"] = (
                    bonuses.get("granary_safety", 0) + GRANARY_SAFETY_BONUS * effectiveness
                )
                bonuses["granary_pest_reduction"] = max(
                    bonuses.get("granary_pest_reduction", 0),
                    effectiveness,  # best granary determines pest reduction
                )

        return bonuses

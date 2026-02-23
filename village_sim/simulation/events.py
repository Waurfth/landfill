"""Random events that add dynamism to the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from numpy.random import Generator

from village_sim.core.config import (
    DISEASE_BASE_PROBABILITY,
    DISCOVERY_PROBABILITY,
    PEST_PROBABILITY,
    PREDATOR_PROBABILITY,
    STORM_PROBABILITY,
)


@dataclass
class Event:
    """A simulation event."""

    event_type: str  # "storm", "disease", "predator", "pest", "discovery", "festival"
    description: str
    day: int
    affected_villager_ids: list[int]
    data: dict


class EventSystem:
    """Generates random events based on conditions."""

    def __init__(self, rng: Generator) -> None:
        self._rng = rng
        self._pending_events: list[Event] = []

    @property
    def pending(self) -> list[Event]:
        return self._pending_events

    def clear_pending(self) -> list[Event]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def check_random_events(
        self,
        day: int,
        season: str,
        weather: str,
        villagers: list["Villager"],  # noqa: F821
    ) -> list[Event]:
        """Generate events for this day. Returns list of events."""
        events: list[Event] = []

        # Storm damage (already handled by weather, but severe storms cause extra)
        if weather == "storm" and self._rng.random() < STORM_PROBABILITY:
            event = Event(
                event_type="storm",
                description="A severe storm struck the village",
                day=day,
                affected_villager_ids=[],
                data={"shelter_damage": self._rng.uniform(0.05, 0.2)},
            )
            events.append(event)

        # Disease outbreak
        if self._rng.random() < DISEASE_BASE_PROBABILITY:
            # Pick a random villager as patient zero
            alive = [v for v in villagers if v.is_alive]
            if alive:
                patient_zero = self._rng.choice(alive)
                # Disease spreads to nearby villagers
                affected = [patient_zero.id]
                for v in alive:
                    if v.id != patient_zero.id:
                        dist = abs(v.current_position[0] - patient_zero.current_position[0]) + \
                               abs(v.current_position[1] - patient_zero.current_position[1])
                        if dist <= 3 and self._rng.random() < 0.3:
                            affected.append(v.id)

                event = Event(
                    event_type="disease",
                    description=f"A sickness spreads through the village, affecting {len(affected)} villagers",
                    day=day,
                    affected_villager_ids=affected,
                    data={"health_damage": self._rng.uniform(10, 30)},
                )
                events.append(event)

        # Predator sighting
        if self._rng.random() < PREDATOR_PROBABILITY:
            event = Event(
                event_type="predator",
                description="A predator has been spotted near the village",
                day=day,
                affected_villager_ids=[],
                data={"danger_increase": 0.1, "duration_days": self._rng.integers(3, 10)},
            )
            events.append(event)

        # Pest infestation (damages stored food)
        if self._rng.random() < PEST_PROBABILITY and season in ("summer", "autumn"):
            event = Event(
                event_type="pest",
                description="Pests have gotten into the food stores",
                day=day,
                affected_villager_ids=[],
                data={"food_loss_fraction": self._rng.uniform(0.05, 0.15)},
            )
            events.append(event)

        # Festival (if conditions are good — enough food, good weather)
        avg_sentiment = sum(v.current_sentiment for v in villagers if v.is_alive) / max(1, len([v for v in villagers if v.is_alive]))
        if (
            weather == "clear"
            and avg_sentiment > 60
            and season in ("spring", "summer")
            and self._rng.random() < 0.02
        ):
            alive = [v for v in villagers if v.is_alive]
            event = Event(
                event_type="festival",
                description="The village holds a celebration!",
                day=day,
                affected_villager_ids=[v.id for v in alive],
                data={"sentiment_boost": 10.0},
            )
            events.append(event)

        self._pending_events.extend(events)
        return events

    def apply_events(
        self,
        events: list[Event],
        villagers_by_id: dict[int, "Villager"],  # noqa: F821
        family_manager: "FamilyManager",  # noqa: F821
        infrastructure: "InfrastructureManager",  # noqa: F821
    ) -> None:
        """Apply event effects to the simulation state."""
        for event in events:
            if event.event_type == "storm":
                # Damage shelters
                damage = event.data["shelter_damage"]
                for s in infrastructure.structures:
                    if s.structure_type == "shelter":
                        s.durability = max(0, s.durability - damage)

            elif event.event_type == "disease":
                health_damage = event.data["health_damage"]
                for vid in event.affected_villager_ids:
                    v = villagers_by_id.get(vid)
                    if v and v.is_alive:
                        v.health = max(0, v.health - health_damage)
                        v.needs.satisfy("health", -health_damage / 100.0)
                        v.memory.add_event(event.day, "fell_sick", -0.3)

            elif event.event_type == "pest":
                loss_frac = event.data["food_loss_fraction"]
                for fam in family_manager.families.values():
                    for item_type in list(fam.inventory.items.keys()):
                        from village_sim.economy.inventory import ITEM_CATALOG
                        if "food_value" in ITEM_CATALOG.get(item_type, {}):
                            for item in fam.inventory.items.get(item_type, []):
                                item.quantity *= (1.0 - loss_frac)

            elif event.event_type == "festival":
                boost = event.data["sentiment_boost"]
                for vid in event.affected_villager_ids:
                    v = villagers_by_id.get(vid)
                    if v and v.is_alive:
                        v.current_sentiment = min(100, v.current_sentiment + boost)
                        v.needs.satisfy("social", 0.3)
                        v.needs.satisfy("purpose", 0.2)
                        v.memory.add_event(event.day, "festival", 0.5)

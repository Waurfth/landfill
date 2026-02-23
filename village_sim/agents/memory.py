"""Agent memory: skills, route familiarity, knowledge, and experiences."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from village_sim.core.config import INTELLIGENCE_LEARNING_BONUS, SKILL_LEARNING_RATE


@dataclass
class Memory:
    """What an agent knows and has experienced."""

    # Skill experience (learning by doing)
    skill_experience: dict[str, float] = field(default_factory=dict)

    # Route familiarity: (start, end) -> trip count
    route_familiarity: dict[tuple[tuple[int, int], tuple[int, int]], int] = field(
        default_factory=dict
    )

    # Knowledge
    known_resource_nodes: list[int] = field(default_factory=list)  # node IDs
    known_recipes: list[str] = field(default_factory=list)
    known_medicinal: list[str] = field(default_factory=list)

    # Social memory: villager_id -> list of (day, event_type, sentiment_change)
    interaction_history: dict[int, list[tuple[int, str, float]]] = field(
        default_factory=dict
    )

    # Recent experiences for sentiment: (day, description, emotional_impact)
    recent_events: deque[tuple[int, str, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    # Yesterday's activity for habit inertia
    last_activity: str | None = None

    def add_experience(self, activity: str, success: bool, intelligence: float = 50.0) -> float:
        """Gain XP from performing an activity. Returns XP gained."""
        xp_gain = 1.0 if success else 0.3
        # Intelligence bonus for learning
        xp_gain *= 1.0 + INTELLIGENCE_LEARNING_BONUS * (intelligence / 100.0)
        self.skill_experience[activity] = self.skill_experience.get(activity, 0.0) + xp_gain
        return xp_gain

    def skill_level(self, activity: str, intelligence: float = 50.0) -> float:
        """Compute skill level (0-100) from XP using diminishing returns."""
        xp = self.skill_experience.get(activity, 0.0)
        # learning_rate modified by intelligence
        effective_rate = SKILL_LEARNING_RATE * (1.0 + 0.5 * (intelligence / 100.0))
        return 100.0 * (1.0 - math.exp(-xp / effective_rate))

    def add_route_trip(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Increment familiarity with a route."""
        key = (start, end)
        self.route_familiarity[key] = self.route_familiarity.get(key, 0) + 1

    def recall_sentiment(self, days: int = 30) -> float:
        """Average emotional impact of recent events."""
        if not self.recent_events:
            return 0.0
        return sum(impact for _, _, impact in self.recent_events) / len(self.recent_events)

    def add_interaction(self, villager_id: int, day: int, event_type: str, sentiment_change: float) -> None:
        """Record an interaction with another villager."""
        if villager_id not in self.interaction_history:
            self.interaction_history[villager_id] = []
        history = self.interaction_history[villager_id]
        history.append((day, event_type, sentiment_change))
        # Keep only last 20 interactions per person
        if len(history) > 20:
            self.interaction_history[villager_id] = history[-20:]

    def add_event(self, day: int, description: str, emotional_impact: float) -> None:
        """Record a recent experience."""
        self.recent_events.append((day, description, emotional_impact))

    def learn_from(
        self,
        other_memory: Memory,
        topic: str,
        own_intelligence: float,
        own_sociability: float,
        relationship_quality: float,
    ) -> bool:
        """Attempt to acquire knowledge from another agent."""
        # Success probability based on intelligence, sociability, relationship
        chance = (
            0.1
            + 0.3 * (own_intelligence / 100)
            + 0.2 * (own_sociability / 100)
            + 0.2 * relationship_quality
        )
        # This returns bool indicating if learning is possible;
        # the actual random roll happens in the caller
        if topic == "recipe":
            new_recipes = [r for r in other_memory.known_recipes if r not in self.known_recipes]
            if new_recipes:
                self.known_recipes.append(new_recipes[0])
                return True
        elif topic == "resource":
            new_nodes = [n for n in other_memory.known_resource_nodes if n not in self.known_resource_nodes]
            if new_nodes:
                self.known_resource_nodes.append(new_nodes[0])
                return True
        elif topic == "medicinal":
            new_med = [m for m in other_memory.known_medicinal if m not in self.known_medicinal]
            if new_med:
                self.known_medicinal.append(new_med[0])
                return True
        return False

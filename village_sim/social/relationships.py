"""Pairwise relationship tracking with asymmetric trust dynamics."""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.core.config import (
    RELATIONSHIP_DECAY_RATE,
    TRUST_GAIN_PER_POSITIVE,
    TRUST_LOSS_PER_NEGATIVE,
)


@dataclass
class Relationship:
    """Pairwise relationship between two villagers."""

    villager_a_id: int
    villager_b_id: int
    trust: float = 0.0        # -1.0 to 1.0
    affinity: float = 0.0     # -1.0 to 1.0 (liking)
    familiarity: float = 0.0  # 0 to 1.0 (how well they know each other)
    interaction_count: int = 0
    last_interaction_day: int = 0

    def positive_interaction(self, magnitude: float = 1.0) -> None:
        """Record a positive interaction. Trust gains are slow."""
        self.trust = min(1.0, self.trust + TRUST_GAIN_PER_POSITIVE * magnitude)
        self.affinity = min(1.0, self.affinity + 0.03 * magnitude)
        self.familiarity = min(1.0, self.familiarity + 0.02)
        self.interaction_count += 1

    def negative_interaction(self, magnitude: float = 1.0) -> None:
        """Record a negative interaction. Trust losses are fast (asymmetric)."""
        self.trust = max(-1.0, self.trust - TRUST_LOSS_PER_NEGATIVE * magnitude)
        self.affinity = max(-1.0, self.affinity - 0.06 * magnitude)
        self.interaction_count += 1

    def daily_decay(self, current_day: int) -> None:
        """Relationships fade without interaction."""
        days_since = current_day - self.last_interaction_day
        if days_since > 0:
            decay = RELATIONSHIP_DECAY_RATE * days_since
            self.affinity *= max(0.0, 1.0 - decay)
            self.familiarity *= max(0.0, 1.0 - decay * 0.5)
            # Trust decays very slowly
            if self.trust > 0:
                self.trust = max(0.0, self.trust - decay * 0.1)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    """Canonical key for a pair of villagers."""
    return (min(a, b), max(a, b))


class RelationshipManager:
    """Manages all pairwise relationships."""

    def __init__(self) -> None:
        self._relationships: dict[tuple[int, int], Relationship] = {}

    def get_or_create(self, a_id: int, b_id: int) -> Relationship:
        key = _pair_key(a_id, b_id)
        if key not in self._relationships:
            self._relationships[key] = Relationship(
                villager_a_id=key[0], villager_b_id=key[1]
            )
        return self._relationships[key]

    def get_all_for(self, villager_id: int) -> list[Relationship]:
        return [
            r for key, r in self._relationships.items()
            if villager_id in key
        ]

    def get_friends(self, villager_id: int, min_affinity: float = 0.3) -> list[int]:
        """Get IDs of villagers this person considers a friend."""
        friends: list[int] = []
        for key, r in self._relationships.items():
            if villager_id in key and r.affinity >= min_affinity:
                other = key[1] if key[0] == villager_id else key[0]
                friends.append(other)
        return friends

    def get_trusted(self, villager_id: int, min_trust: float = 0.3) -> list[int]:
        """Get IDs of villagers this person trusts."""
        trusted: list[int] = []
        for key, r in self._relationships.items():
            if villager_id in key and r.trust >= min_trust:
                other = key[1] if key[0] == villager_id else key[0]
                trusted.append(other)
        return trusted

    def strongest_relationships(
        self, villager_id: int, n: int = 5
    ) -> list[Relationship]:
        rels = self.get_all_for(villager_id)
        rels.sort(key=lambda r: r.affinity + r.trust, reverse=True)
        return rels[:n]

    def daily_decay_all(self, current_day: int) -> None:
        """Decay all relationships that haven't been interacted with."""
        for r in self._relationships.values():
            r.daily_decay(current_day)

    def record_interaction(
        self, a_id: int, b_id: int, positive: bool, magnitude: float, day: int
    ) -> None:
        """Record an interaction between two villagers."""
        rel = self.get_or_create(a_id, b_id)
        rel.last_interaction_day = day
        if positive:
            rel.positive_interaction(magnitude)
        else:
            rel.negative_interaction(magnitude)

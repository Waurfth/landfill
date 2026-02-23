"""Maslow-inspired need system with decay rates and urgency curves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from village_sim.core.config import (
    COMFORT_DECAY_RATE,
    HUNGER_DECAY_RATE,
    NEED_WEIGHTS,
    PURPOSE_DECAY_RATE,
    REST_DECAY_RATE,
    SAFETY_DECAY_RATE,
    SHELTER_DECAY_RATE,
    SOCIAL_DECAY_RATE,
    SURVIVAL_CRITICAL_THRESHOLD,
    THIRST_DECAY_RATE,
    WARMTH_DECAY_RATE,
)


@dataclass
class Need:
    """A single need with satisfaction level and urgency."""

    name: str
    satisfaction: float = 1.0   # 0.0 (desperate) to 1.0 (fully satisfied)
    decay_rate: float = 0.1     # per day base decay
    weight: float = 1.0         # importance in decision making
    urgency_curve: str = "exponential"  # "exponential" or "linear"

    def decay(self, modifier: float = 1.0) -> None:
        """Decay satisfaction by one day, applying modifier."""
        self.satisfaction -= self.decay_rate * modifier
        self.satisfaction = max(0.0, self.satisfaction)

    def urgency(self) -> float:
        """Urgency score: higher when satisfaction is lower."""
        deficit = 1.0 - self.satisfaction
        if self.urgency_curve == "exponential":
            # Exponential ramp: urgency explodes as satisfaction approaches 0
            return self.weight * (math.exp(deficit * 3) - 1) / (math.exp(3) - 1)
        # Linear
        return self.weight * deficit

    def satisfy(self, amount: float) -> None:
        """Increase satisfaction by amount."""
        self.satisfaction = min(1.0, self.satisfaction + amount)

    @property
    def is_critical(self) -> bool:
        return self.satisfaction < SURVIVAL_CRITICAL_THRESHOLD


# Names of needs that are survival-critical
_SURVIVAL_NEEDS = {"hunger", "thirst", "rest", "health", "warmth"}


class NeedSystem:
    """Manages all needs for a villager."""

    def __init__(self) -> None:
        self.needs: dict[str, Need] = {
            "hunger": Need("hunger", 1.0, HUNGER_DECAY_RATE, NEED_WEIGHTS["hunger"], "exponential"),
            "thirst": Need("thirst", 1.0, THIRST_DECAY_RATE, NEED_WEIGHTS["thirst"], "exponential"),
            "rest": Need("rest", 1.0, REST_DECAY_RATE, NEED_WEIGHTS["rest"], "exponential"),
            "warmth": Need("warmth", 1.0, WARMTH_DECAY_RATE, NEED_WEIGHTS["warmth"], "exponential"),
            "shelter": Need("shelter", 1.0, SHELTER_DECAY_RATE, NEED_WEIGHTS["shelter"], "linear"),
            "safety": Need("safety", 1.0, SAFETY_DECAY_RATE, NEED_WEIGHTS["safety"], "linear"),
            "health": Need("health", 1.0, 0.0, NEED_WEIGHTS["health"], "exponential"),  # no natural decay
            "social": Need("social", 1.0, SOCIAL_DECAY_RATE, NEED_WEIGHTS["social"], "linear"),
            "purpose": Need("purpose", 1.0, PURPOSE_DECAY_RATE, NEED_WEIGHTS["purpose"], "linear"),
            "comfort": Need("comfort", 0.5, COMFORT_DECAY_RATE, NEED_WEIGHTS["comfort"], "linear"),
        }

    def get_most_urgent(self) -> Need:
        """Return the need with highest urgency * weight."""
        return max(self.needs.values(), key=lambda n: n.urgency())

    def get_urgency_vector(self) -> dict[str, float]:
        """Map of need_name -> urgency score."""
        return {name: need.urgency() for name, need in self.needs.items()}

    def satisfy(self, need_name: str, amount: float) -> None:
        """Satisfy a specific need."""
        if need_name in self.needs:
            self.needs[need_name].satisfy(amount)

    def daily_decay(
        self,
        warmth_modifier: float = 1.0,
        shelter_quality: float = 0.0,
        had_social_interaction: bool = False,
        was_productive: bool = False,
    ) -> None:
        """Decay all needs for one day with situational modifiers."""
        for name, need in self.needs.items():
            modifier = 1.0
            if name == "warmth":
                modifier = warmth_modifier
                # Good shelter reduces warmth decay
                modifier *= max(0.3, 1.0 - shelter_quality * 0.7)
            elif name == "shelter":
                modifier = max(0.1, 1.0 - shelter_quality)
            elif name == "social":
                if had_social_interaction:
                    modifier = 0.0  # no decay if they socialized
            elif name == "purpose":
                if was_productive:
                    modifier = 0.0
            elif name == "health":
                continue  # health doesn't decay naturally

            need.decay(modifier)

    def overall_wellbeing(self) -> float:
        """Weighted average of all satisfactions (0-1)."""
        total_weight = sum(n.weight for n in self.needs.values())
        if total_weight == 0:
            return 0.5
        weighted = sum(n.satisfaction * n.weight for n in self.needs.values())
        return weighted / total_weight

    def survival_critical(self) -> bool:
        """Are any survival needs below the critical threshold?"""
        return any(
            self.needs[name].is_critical
            for name in _SURVIVAL_NEEDS
            if name in self.needs
        )

    def most_urgent_survival(self) -> Need | None:
        """Return the most urgent survival need, or None if none are urgent."""
        survival = [
            self.needs[name]
            for name in _SURVIVAL_NEEDS
            if name in self.needs and self.needs[name].satisfaction < 0.5
        ]
        if not survival:
            return None
        return max(survival, key=lambda n: n.urgency())

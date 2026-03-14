"""Agent memory: skills, route familiarity, knowledge, and experiences.

Three-tier episodic memory system:
  Tier 1 — Episodes: detailed recent events with decaying salience (max 40)
  Tier 2 — Impressions: compressed permanent summaries per event category
  Tier 3 — Activity Biases: learned avoidance/attraction for activities (±0.3)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from village_sim.core.config import (
    ACTIVITY_BIAS_SPREAD,
    INTELLIGENCE_LEARNING_BONUS,
    MAX_EPISODES_PER_VILLAGER,
    MEMORY_BIAS_DECAY_RATE,
    MEMORY_BIAS_LEARNING_RATE,
    MEMORY_BIAS_MAX,
    MEMORY_CONSOLIDATION_THRESHOLD,
    MEMORY_IMPRESSION_WEIGHT,
    MEMORY_SALIENCE_DECAY,
    MEMORY_TRAUMATIC_THRESHOLD,
    SKILL_LEARNING_RATE,
)


# =========================================================================
# Tier 1: Episodes
# =========================================================================

@dataclass
class Episode:
    """A single significant event in a villager's life."""

    day: int
    category: str           # e.g. "injury", "death_of_kin", "marriage"
    description: str        # human-readable
    emotional_impact: float  # -1.0 (devastating) to +1.0 (euphoric)
    activity: str = ""      # activity during which it happened (if any)
    other_villager_id: int = -1  # related villager (-1 if none)
    salience: float = 1.0   # decays daily; consolidated when low


# =========================================================================
# Tier 2: Impressions
# =========================================================================

@dataclass
class Impression:
    """Compressed summary of all episodes in a category."""

    category: str
    count: int = 0
    avg_emotional_impact: float = 0.0
    most_recent_day: int = 0
    peak_impact: float = 0.0   # strongest single impact (signed)

    def absorb(self, episode: Episode) -> None:
        """Merge an episode into this impression."""
        total = self.avg_emotional_impact * self.count + episode.emotional_impact
        self.count += 1
        self.avg_emotional_impact = total / self.count
        self.most_recent_day = max(self.most_recent_day, episode.day)
        if abs(episode.emotional_impact) > abs(self.peak_impact):
            self.peak_impact = episode.emotional_impact


# =========================================================================
# Experiential trait modifier mappings
# =========================================================================

# category -> (trait_name, weight_per_count, cap)
# Positive weight = episodes with positive impact raise the trait;
# episodes with negative impact lower it.
_EXPERIENTIAL_RULES: dict[str, list[tuple[str, float, float]]] = {
    "festival":           [("baseline_optimism", 0.5, 10.0), ("sociability", 0.3, 5.0)],
    "death_of_kin":       [("baseline_optimism", -1.5, 10.0)],
    "death_of_friend":    [("baseline_optimism", -0.8, 10.0)],
    "injury":             [("risk_tolerance", -1.0, 8.0)],
    "near_death":         [("risk_tolerance", -2.0, 8.0)],
    "predator_encounter": [("risk_tolerance", -0.6, 8.0)],
    "successful_hunt":    [("risk_tolerance", 0.4, 8.0)],
    "marriage":           [("sociability", 1.0, 5.0)],
    "trade_success":      [("sociability", 0.2, 5.0)],
    "trade_betrayal":     [("sociability", -0.5, 5.0)],
    "starvation":         [("loss_aversion", 1.5, 8.0)],
    "healed":             [("empathy", 0.5, 5.0)],
    "birth_of_child":     [("empathy", 0.8, 5.0), ("baseline_optimism", 0.5, 10.0)],
}


# =========================================================================
# Memory (extended)
# =========================================================================

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

    # Legacy recent experiences (kept for backward compat)
    recent_events: deque[tuple[int, str, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    # Yesterday's activity for habit inertia
    last_activity: str | None = None

    # --- Tier 1: Episodes ---
    episodes: list[Episode] = field(default_factory=list)

    # --- Tier 2: Impressions ---
    impressions: dict[str, Impression] = field(default_factory=dict)

    # --- Tier 3: Activity biases ---
    activity_biases: dict[str, float] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Skills
    # ----------------------------------------------------------------

    def add_experience(self, activity: str, success: bool, intelligence: float = 50.0) -> float:
        """Gain XP from performing an activity. Returns XP gained."""
        xp_gain = 1.0 if success else 0.3
        xp_gain *= 1.0 + INTELLIGENCE_LEARNING_BONUS * (intelligence / 100.0)
        self.skill_experience[activity] = self.skill_experience.get(activity, 0.0) + xp_gain
        return xp_gain

    def skill_level(self, activity: str, intelligence: float = 50.0) -> float:
        """Compute skill level (0-100) from XP using diminishing returns."""
        xp = self.skill_experience.get(activity, 0.0)
        effective_rate = SKILL_LEARNING_RATE * (1.0 + 0.5 * (intelligence / 100.0))
        return 100.0 * (1.0 - math.exp(-xp / effective_rate))

    # ----------------------------------------------------------------
    # Routes
    # ----------------------------------------------------------------

    def add_route_trip(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Increment familiarity with a route."""
        key = (start, end)
        self.route_familiarity[key] = self.route_familiarity.get(key, 0) + 1

    # ----------------------------------------------------------------
    # Episodes (Tier 1)
    # ----------------------------------------------------------------

    def add_episode(
        self,
        day: int,
        category: str,
        description: str,
        emotional_impact: float,
        activity: str = "",
        other_villager_id: int = -1,
    ) -> None:
        """Record a significant life event as an episode.

        Also updates activity biases (Tier 3) and feeds the legacy
        recent_events deque for backward compatibility.
        """
        emotional_impact = max(-1.0, min(1.0, emotional_impact))
        ep = Episode(
            day=day,
            category=category,
            description=description,
            emotional_impact=emotional_impact,
            activity=activity,
            other_villager_id=other_villager_id,
        )
        self.episodes.append(ep)

        # Legacy compat
        self.recent_events.append((day, description, emotional_impact))

        # Update activity bias (Tier 3)
        if activity:
            self._update_activity_bias(activity, emotional_impact)

        # Force-consolidate if over cap
        if len(self.episodes) > MAX_EPISODES_PER_VILLAGER:
            self._force_consolidate_oldest()

    def _update_activity_bias(self, activity: str, emotional_impact: float) -> None:
        """Shift activity bias based on episode impact."""
        delta = emotional_impact * MEMORY_BIAS_LEARNING_RATE
        current = self.activity_biases.get(activity, 0.0)
        self.activity_biases[activity] = max(
            -MEMORY_BIAS_MAX, min(MEMORY_BIAS_MAX, current + delta)
        )
        # Spread to related activities
        for related, strength in ACTIVITY_BIAS_SPREAD.get(activity, []):
            cur = self.activity_biases.get(related, 0.0)
            self.activity_biases[related] = max(
                -MEMORY_BIAS_MAX, min(MEMORY_BIAS_MAX, cur + delta * strength)
            )

    def _force_consolidate_oldest(self) -> None:
        """Consolidate the lowest-salience episode to stay under cap."""
        if not self.episodes:
            return
        # Find lowest salience
        worst_idx = min(range(len(self.episodes)), key=lambda i: self.episodes[i].salience)
        ep = self.episodes.pop(worst_idx)
        self._consolidate_episode(ep)

    def _consolidate_episode(self, ep: Episode) -> None:
        """Merge a single episode into its category impression (Tier 2)."""
        if ep.category not in self.impressions:
            self.impressions[ep.category] = Impression(category=ep.category)
        self.impressions[ep.category].absorb(ep)

    # ----------------------------------------------------------------
    # Daily consolidation
    # ----------------------------------------------------------------

    def consolidate_memories(self, emotional_stability: float = 50.0) -> None:
        """Daily memory maintenance: decay salience and consolidate faded episodes.

        *emotional_stability* (0-100 trait) affects how quickly salience fades.
        Stable personalities forget faster (move on); volatile ones hold on.
        """
        # Stability adjusts decay: high stability -> faster decay
        stability_factor = 0.5 + 0.5 * (emotional_stability / 100.0)
        decay = MEMORY_SALIENCE_DECAY ** stability_factor  # e.g. 0.97^0.75

        to_consolidate: list[int] = []
        for i, ep in enumerate(self.episodes):
            ep.salience *= decay
            # Traumatic events resist consolidation
            if abs(ep.emotional_impact) >= MEMORY_TRAUMATIC_THRESHOLD:
                threshold = MEMORY_CONSOLIDATION_THRESHOLD * 0.5
            else:
                threshold = MEMORY_CONSOLIDATION_THRESHOLD
            if ep.salience < threshold:
                to_consolidate.append(i)

        # Remove from end to preserve indices
        for i in reversed(to_consolidate):
            ep = self.episodes.pop(i)
            self._consolidate_episode(ep)

    def decay_biases(self) -> None:
        """Daily decay of activity biases toward zero."""
        to_remove: list[str] = []
        for act, bias in self.activity_biases.items():
            if bias > 0:
                bias = max(0.0, bias - MEMORY_BIAS_DECAY_RATE)
            else:
                bias = min(0.0, bias + MEMORY_BIAS_DECAY_RATE)
            if abs(bias) < 0.001:
                to_remove.append(act)
            else:
                self.activity_biases[act] = bias
        for act in to_remove:
            del self.activity_biases[act]

    # ----------------------------------------------------------------
    # Queries
    # ----------------------------------------------------------------

    def get_activity_bias(self, activity_name: str) -> float:
        """Lookup learned bias for an activity (Tier 3)."""
        return self.activity_biases.get(activity_name, 0.0)

    def get_experiential_modifier(self, trait_name: str) -> float:
        """Compute trait shift from accumulated impressions (Tier 2).

        Returns a value to add to the trait's base (typically ±0-10).
        """
        modifier = 0.0
        for category, imp in self.impressions.items():
            rules = _EXPERIENTIAL_RULES.get(category, [])
            for rule_trait, weight, cap in rules:
                if rule_trait == trait_name:
                    # Contribution scales with count (diminishing) and avg impact
                    contribution = weight * math.sqrt(imp.count) * abs(imp.avg_emotional_impact)
                    # Sign: weight sign determines direction
                    if weight < 0:
                        contribution = -contribution
                    modifier += max(-cap, min(cap, contribution))
        return modifier

    def get_episodes_involving(self, villager_id: int) -> list[Episode]:
        """Get all current episodes involving a specific villager."""
        return [ep for ep in self.episodes if ep.other_villager_id == villager_id]

    def recall_sentiment(self, days: int = 30) -> float:
        """Blended sentiment from episodes (Tier 1) and impressions (Tier 2).

        Episodes weighted by salience; impressions provide a background mood.
        """
        # Episode-based (salience-weighted)
        ep_total = 0.0
        ep_weight = 0.0
        for ep in self.episodes:
            ep_total += ep.emotional_impact * ep.salience
            ep_weight += ep.salience

        ep_sentiment = ep_total / ep_weight if ep_weight > 0 else 0.0

        # Impression background mood
        imp_total = 0.0
        imp_count = 0
        for imp in self.impressions.values():
            if imp.count > 0:
                imp_total += imp.avg_emotional_impact
                imp_count += 1
        imp_mood = (imp_total / imp_count) if imp_count > 0 else 0.0

        # Blend: episodes dominate when present, impressions provide baseline
        if ep_weight > 0:
            return ep_sentiment * 0.8 + imp_mood * MEMORY_IMPRESSION_WEIGHT
        if imp_count > 0:
            return imp_mood * MEMORY_IMPRESSION_WEIGHT
        # Fallback to legacy
        if self.recent_events:
            return sum(impact for _, _, impact in self.recent_events) / len(self.recent_events)
        return 0.0

    # ----------------------------------------------------------------
    # Legacy / backward compat
    # ----------------------------------------------------------------

    def add_event(self, day: int, description: str, emotional_impact: float) -> None:
        """Record a recent experience (legacy interface).

        Maps to add_episode with category inferred from description.
        """
        category = _infer_category(description)
        activity = _infer_activity(description)
        self.add_episode(day, category, description, emotional_impact, activity=activity)

    def add_interaction(self, villager_id: int, day: int, event_type: str, sentiment_change: float) -> None:
        """Record an interaction with another villager."""
        if villager_id not in self.interaction_history:
            self.interaction_history[villager_id] = []
        history = self.interaction_history[villager_id]
        history.append((day, event_type, sentiment_change))
        if len(history) > 20:
            self.interaction_history[villager_id] = history[-20:]

    def learn_from(
        self,
        other_memory: Memory,
        topic: str,
        own_intelligence: float,
        own_sociability: float,
        relationship_quality: float,
    ) -> bool:
        """Attempt to acquire knowledge from another agent."""
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


# =========================================================================
# Helpers
# =========================================================================

def _infer_category(description: str) -> str:
    """Best-effort category from a legacy description string."""
    d = description.lower()
    if "injured" in d:
        return "injury"
    if "sick" in d or "disease" in d or "ill" in d:
        return "illness"
    if "festival" in d or "celebrat" in d:
        return "festival"
    if "died" in d or "death" in d:
        return "death_of_kin"
    if "born" in d or "birth" in d:
        return "birth_of_child"
    if "married" in d or "wedding" in d:
        return "marriage"
    if "storm" in d:
        return "storm_damage"
    if "predator" in d:
        return "predator_encounter"
    if "trade" in d:
        return "trade_success"
    if "starv" in d or "hunger" in d:
        return "starvation"
    return "misc"


def _infer_activity(description: str) -> str:
    """Best-effort activity from a legacy description string."""
    d = description.lower()
    if "during " in d:
        # "injured during hunt_large_game" -> "hunt_large_game"
        return d.split("during ")[-1].strip()
    return ""

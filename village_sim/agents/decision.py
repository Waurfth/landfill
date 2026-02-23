"""Multi-activity daily decision engine with personality-driven heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from numpy.random import Generator

from village_sim.core.config import (
    FATIGUE_STOP_THRESHOLD,
    HABIT_INERTIA_BONUS,
    SATISFICE_THRESHOLD,
)
from village_sim.economy.activities import ACTIVITIES, ACTIVITY_NEED_MAPPING, Activity


@dataclass
class ActivityPlan:
    """A planned activity for a time slot in the day."""

    activity_name: str
    target_resource_id: Optional[int] = None
    target_villager_id: Optional[int] = None
    planned_hours: float = 4.0
    target_position: Optional[tuple[int, int]] = None


@dataclass
class SocialAction:
    """A social interaction decision."""

    action_type: str  # "chat", "share_food", "propose_work", "court", "teach"
    target_id: int = -1
    topic: str = ""


class DecisionEngine:
    """Heuristic decision-making: personality-driven, satisficing, habit-forming."""

    def __init__(self, rng: Generator) -> None:
        self._rng = rng

    def plan_day(
        self,
        villager: "Villager",  # noqa: F821
        world_state: "WorldState",  # noqa: F821
        available_hours: float,
    ) -> list[ActivityPlan]:
        """
        Plan a full day of activities, filling available hours.

        Process:
        1. Check survival needs -> if critical, address those
        2. Pick most urgent need, map to activities
        3. Evaluate candidate activities
        4. Apply personality biases and habit inertia
        5. Satisfice: pick first "good enough" option
        6. Subtract hours, repeat
        """
        schedule: list[ActivityPlan] = []
        remaining = available_hours

        while remaining > 1.0:
            plan = self._pick_next_activity(villager, world_state, remaining, schedule)
            if plan is None:
                # No viable activity — rest
                schedule.append(ActivityPlan("rest", planned_hours=remaining))
                break

            schedule.append(plan)
            remaining -= plan.planned_hours

            # Check fatigue
            act = ACTIVITIES.get(plan.activity_name)
            if act and villager.fatigue + act.fatigue_cost * plan.planned_hours > FATIGUE_STOP_THRESHOLD:
                break

        # Update habit memory
        if schedule:
            villager.memory.last_activity = schedule[0].activity_name

        return schedule

    def _pick_next_activity(
        self,
        villager: "Villager",  # noqa: F821
        world_state: "WorldState",  # noqa: F821
        remaining_hours: float,
        current_schedule: list[ActivityPlan],
    ) -> Optional[ActivityPlan]:
        """Pick the best activity for the next time slot."""
        # Get urgency vector
        urgencies = villager.needs.get_urgency_vector()

        # Determine target needs
        if villager.needs.survival_critical():
            # Survival mode: only consider survival-satisfying activities
            target_needs = _survival_needs(urgencies)
        else:
            # Normal mode: consider all needs, weighted by urgency
            target_needs = sorted(urgencies.keys(), key=lambda n: urgencies[n], reverse=True)

        # Generate candidate activities
        candidates: list[tuple[float, ActivityPlan]] = []

        for need_name in target_needs[:5]:  # top 5 needs
            activities_for_need = _activities_for_need(need_name)
            for act_name in activities_for_need:
                act = ACTIVITIES.get(act_name)
                if act is None:
                    continue

                plan = self._evaluate_activity(
                    villager, act, world_state, remaining_hours, urgencies, current_schedule,
                )
                if plan is not None:
                    score, activity_plan = plan
                    candidates.append((score, activity_plan))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Satisficing: take first "good enough" option, with some randomness
        # from personality
        for score, plan in candidates:
            if score >= SATISFICE_THRESHOLD:
                return plan
            # Lower-ambition villagers accept lower scores
            ambition = villager.traits.ambition
            adjusted_threshold = SATISFICE_THRESHOLD * (0.7 + 0.6 * ambition / 100)
            if score >= adjusted_threshold:
                return plan

        # Nothing good enough — take the best anyway
        return candidates[0][1]

    def _evaluate_activity(
        self,
        villager: "Villager",  # noqa: F821
        activity: Activity,
        world_state: "WorldState",  # noqa: F821
        remaining_hours: float,
        urgencies: dict[str, float],
        current_schedule: list[ActivityPlan],
    ) -> Optional[tuple[float, ActivityPlan]]:
        """
        Evaluate a candidate activity. Returns (score, plan) or None if infeasible.
        """
        # Check time feasibility
        if activity.base_hours > remaining_hours and activity.base_hours > 0:
            return None

        # Check season
        if activity.required_season is not None:
            if world_state.season not in activity.required_season:
                return None

        # Check tools — search both personal and family inventory
        inv = villager.personal_inventory
        family_inv = world_state.get_family_inventory(villager.family_id)

        tool_quality = 1.0
        if activity.required_tools:
            found_tool = False
            for tool_type in activity.required_tools:
                tool = None
                if inv is not None:
                    tool = inv.get_best_tool(tool_type)
                if tool is None and family_inv is not None:
                    tool = family_inv.get_best_tool(tool_type)
                if tool is not None:
                    found_tool = True
                    tool_quality = tool.tool_quality
                    break
            if not found_tool:
                return None

        # Find resource node
        target_resource_id = None
        target_pos = None
        travel_hours = 0.0
        if activity.resource_type is not None:
            node = world_state.find_resource(
                villager.current_position, activity.resource_type
            )
            if node is None or node.current_abundance <= 0:
                return None
            target_resource_id = node.node_id
            target_pos = node.position
            travel_hours = world_state.estimate_travel(
                villager.current_position, node.position
            )
            if travel_hours * 2 + activity.base_hours > remaining_hours:
                return None

        # Calculate base score
        score = 0.0

        # Need satisfaction potential
        satisfied_needs = ACTIVITY_NEED_MAPPING.get(activity.name, [])
        for need_name in satisfied_needs:
            score += urgencies.get(need_name, 0.0) * 0.3

        # Success probability
        success_prob = activity.calculate_success(
            villager, tool_quality,
            weather_modifier=world_state.weather_modifier,
        )
        score *= success_prob

        # Risk aversion
        risk_tolerance = villager.traits.risk_tolerance / 100.0
        risk_penalty = activity.danger_level * (1.5 - risk_tolerance)
        score -= risk_penalty

        # Time efficiency (prefer shorter activities when many things to do)
        if activity.base_hours > 0:
            efficiency = 1.0 / (activity.base_hours + travel_hours * 2)
            score += efficiency * 0.1

        # Personality biases
        score = self._apply_personality_biases(
            villager, activity, score, current_schedule
        )

        # Habit inertia
        if villager.memory.last_activity == activity.name:
            score += HABIT_INERTIA_BONUS

        # Intelligence noise (low intelligence = more random choices)
        noise = self._rng.uniform(-0.1, 0.1) * (1.0 - villager.traits.intelligence / 100.0)
        score += noise

        planned_hours = min(activity.base_hours or remaining_hours, remaining_hours)
        if travel_hours > 0:
            planned_hours = min(planned_hours, remaining_hours - travel_hours * 2)

        plan = ActivityPlan(
            activity_name=activity.name,
            target_resource_id=target_resource_id,
            planned_hours=max(1.0, planned_hours),
            target_position=target_pos,
        )
        return (max(0.0, score), plan)

    def _apply_personality_biases(
        self,
        villager: "Villager",  # noqa: F821
        activity: Activity,
        score: float,
        current_schedule: list[ActivityPlan],
    ) -> float:
        """Apply personality-specific biases to activity score."""
        # Patient villagers like farming and fishing
        if activity.name in ("fishing", "farm_tend", "farm_plant"):
            score += (villager.traits.patience - 50) / 100.0 * 0.15

        # Ambitious villagers prefer high-value activities
        if activity.name in ("hunt_large_game", "mine_ore", "craft_tools"):
            score += (villager.traits.ambition - 50) / 100.0 * 0.15

        # Social villagers prefer group activities
        if activity.min_group_size > 1:
            score += (villager.traits.sociability - 50) / 100.0 * 0.10

        # Creative villagers prefer crafting
        if activity.name in ("craft_tools", "cook_food"):
            score += (villager.traits.creativity - 50) / 100.0 * 0.10

        # Conscientious villagers prefer farming (consistent, reliable work)
        if activity.name.startswith("farm_"):
            score += (villager.traits.conscientiousness - 50) / 100.0 * 0.10

        # Risk-tolerant villagers like dangerous activities
        if activity.danger_level > 0.05:
            score += (villager.traits.risk_tolerance - 50) / 100.0 * 0.10

        # Impatient villagers prefer quick activities
        if activity.base_hours <= 3:
            impatience = (100 - villager.traits.patience) / 100.0
            score += impatience * 0.08

        return score

    def decide_social(
        self,
        villager: "Villager",  # noqa: F821
        available_villagers: list["Villager"],  # noqa: F821
        relationships: "RelationshipManager",  # noqa: F821
    ) -> Optional[SocialAction]:
        """Decide on an evening social interaction."""
        if not available_villagers:
            return None

        # Sociability drives engagement
        if self._rng.random() > (villager.traits.sociability / 100.0) * 0.8 + 0.2:
            return None

        # Pick target: prefer family, then friends, then random
        friends = relationships.get_friends(villager.id)
        family_nearby = [v for v in available_villagers if v.family_id == villager.family_id]

        if family_nearby and self._rng.random() < 0.4:
            target = self._rng.choice(family_nearby)
        elif friends:
            friend_nearby = [v for v in available_villagers if v.id in friends]
            if friend_nearby:
                target = self._rng.choice(friend_nearby)
            else:
                target = self._rng.choice(available_villagers)
        else:
            target = self._rng.choice(available_villagers)

        # Choose action type
        if villager.needs.needs["social"].satisfaction < 0.3:
            action = "chat"
        elif villager.needs.needs["hunger"].satisfaction < 0.5 and villager.traits.empathy > 50:
            action = "share_food"
        elif villager.is_fertile and target.sex != villager.sex:
            rel = relationships.get_or_create(villager.id, target.id)
            if rel.affinity > 0.4:
                action = "court"
            else:
                action = "chat"
        else:
            action = self._rng.choice(["chat", "teach", "propose_work"])

        return SocialAction(action_type=action, target_id=target.id)

    def evaluate_cooperation_request(
        self,
        villager: "Villager",  # noqa: F821
        proposer: "Villager",  # noqa: F821
        activity_name: str,
        trust: float,
    ) -> bool:
        """Evaluate whether to join a work party."""
        # Is the activity aligned with my needs?
        satisfied_needs = ACTIVITY_NEED_MAPPING.get(activity_name, [])
        urgencies = villager.needs.get_urgency_vector()
        need_alignment = sum(urgencies.get(n, 0) for n in satisfied_needs)

        # Trust in proposer
        trust_factor = 0.3 + 0.7 * max(0, trust)

        # Am I better off solo? (sociable villagers more likely to join)
        solo_preference = (100 - villager.traits.sociability) / 100.0 * 0.3

        score = need_alignment * trust_factor - solo_preference

        return score > 0.3


# =============================================================================
# Helper: WorldState wrapper (lightweight access to world state for decisions)
# =============================================================================

class WorldState:
    """Read-only view of world state for decision-making."""

    def __init__(
        self,
        season: str,
        weather_modifier: float,
        resource_manager: "ResourceManager",  # noqa: F821
        world_map: "WorldMap",  # noqa: F821
        family_inventories: dict[int, "Inventory"],  # noqa: F821
    ) -> None:
        self.season = season
        self.weather_modifier = weather_modifier
        self._resource_manager = resource_manager
        self._world_map = world_map
        self._family_inventories = family_inventories

    def find_resource(
        self, position: tuple[int, int], resource_type: "ResourceType"  # noqa: F821
    ) -> Optional["ResourceNode"]:  # noqa: F821
        return self._resource_manager.get_nearest_of_type(position, resource_type)

    def estimate_travel(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> float:
        """Quick travel time estimate (Manhattan distance / speed)."""
        from village_sim.core.config import BASE_TRAVEL_SPEED
        dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return dist / BASE_TRAVEL_SPEED

    def get_family_inventory(self, family_id: int) -> Optional["Inventory"]:  # noqa: F821
        return self._family_inventories.get(family_id)


def _survival_needs(urgencies: dict[str, float]) -> list[str]:
    """Return only survival-critical need names, sorted by urgency."""
    survival = {"hunger", "thirst", "rest", "health", "warmth"}
    return sorted(
        [n for n in urgencies if n in survival],
        key=lambda n: urgencies[n],
        reverse=True,
    )


def _activities_for_need(need_name: str) -> list[str]:
    """Return activity names that can satisfy a given need."""
    return [
        act_name
        for act_name, needs in ACTIVITY_NEED_MAPPING.items()
        if need_name in needs
    ]

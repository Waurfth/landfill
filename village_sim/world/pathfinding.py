"""A* pathfinding with agent-specific noise and route caching."""

from __future__ import annotations

import heapq
from typing import Optional

from numpy.random import Generator

from village_sim.core.config import BASE_TRAVEL_SPEED, FATIGUE_PER_TRAVEL_HOUR


# Module-level route cache: (start, end) -> (path, cost)
_route_cache: dict[tuple[tuple[int, int], tuple[int, int]], tuple[list[tuple[int, int]], float]] = {}
_CACHE_MAX_SIZE: int = 2000


def clear_cache() -> None:
    """Clear the pathfinding route cache."""
    _route_cache.clear()


def find_path(
    world_map: "WorldMap",  # noqa: F821
    start: tuple[int, int],
    end: tuple[int, int],
    agent: Optional["Villager"] = None,  # noqa: F821
    rng: Optional[Generator] = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    A* pathfinding from start to end.

    Returns:
        (path, total_cost, estimated_hours)
    """
    if start == end:
        return [start], 0.0, 0.0

    # Check cache for the optimal path (no agent noise)
    cache_key = (start, end)
    if agent is None and cache_key in _route_cache:
        path, cost = _route_cache[cache_key]
        hours = cost / BASE_TRAVEL_SPEED
        return list(path), cost, hours

    # Agent-specific noise factor
    noise_scale = 0.0
    if agent is not None and rng is not None:
        intelligence = getattr(agent.traits, "intelligence", 50)
        familiarity = agent.memory.route_familiarity.get(cache_key, 0)
        # Less noise with higher intelligence and more familiarity
        noise_scale = max(0.0, 0.3 * (1 - intelligence / 100) * (1 / (1 + familiarity * 0.5)))

    # A* search
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path: list[tuple[int, int]] = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            total_cost = g_score[end]
            hours = total_cost / BASE_TRAVEL_SPEED
            if agent is not None:
                hours = _adjust_travel_time(hours, agent)
            # Cache optimal paths (agent-free)
            if agent is None and len(_route_cache) < _CACHE_MAX_SIZE:
                _route_cache[cache_key] = (list(path), total_cost)
            return path, total_cost, hours

        for nx, ny in world_map.neighbors(*current):
            move_cost = world_map.movement_cost(nx, ny)
            # Add noise for agents
            if noise_scale > 0 and rng is not None:
                move_cost *= 1.0 + rng.uniform(-noise_scale, noise_scale * 2)
                move_cost = max(0.1, move_cost)

            tentative = g_score[current] + move_cost
            neighbor = (nx, ny)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                h = _heuristic(neighbor, end)
                counter += 1
                heapq.heappush(open_set, (tentative + h, counter, neighbor))

    # No path found
    return [], float("inf"), float("inf")


def estimate_travel_time(
    world_map: "WorldMap",  # noqa: F821
    start: tuple[int, int],
    end: tuple[int, int],
    agent: "Villager",  # noqa: F821
    rng: Optional[Generator] = None,
) -> float:
    """Estimate travel time in hours accounting for agent capabilities."""
    _, _, hours = find_path(world_map, start, end, agent, rng)
    return hours


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _adjust_travel_time(base_hours: float, agent: "Villager") -> float:  # noqa: F821
    """Adjust travel time based on agent's physical state."""
    endurance = getattr(agent, "effective_endurance", 50)
    health = getattr(agent, "health", 100)
    fatigue = getattr(agent, "fatigue", 0)
    age_mod = getattr(agent, "_age_physical_modifier", lambda: 1.0)()

    # Load fraction (how much of carry capacity is used)
    inv = getattr(agent, "personal_inventory", None)
    load_fraction = 0.0
    if inv is not None:
        from village_sim.core.config import CARRY_CAPACITY_BASE
        strength = getattr(agent, "effective_strength", 50)
        capacity = CARRY_CAPACITY_BASE * (strength / 50.0)
        if capacity > 0:
            load_fraction = min(1.0, inv.total_weight() / capacity)

    effective_speed_mult = (
        (endurance / 100.0)
        * (health / 100.0)
        * age_mod
        * (1.0 - load_fraction * 0.5)
        * (1.0 - fatigue * 0.3)
    )
    effective_speed_mult = max(0.2, effective_speed_mult)
    return base_hours / effective_speed_mult

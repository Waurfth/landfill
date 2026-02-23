"""Data collection, statistics, inequality measures, and export."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DailySnapshot:
    """A snapshot of simulation state for one day."""

    day: int = 0
    population: int = 0
    births: int = 0
    deaths: int = 0
    avg_sentiment: float = 0.0
    avg_hunger: float = 0.0
    avg_health: float = 0.0
    total_food: float = 0.0
    food_per_capita: float = 0.0
    gini: float = 0.0
    activity_counts: dict[str, int] = field(default_factory=dict)
    avg_wellbeing: float = 0.0
    marriage_count: int = 0
    trade_count: int = 0
    trade_items_exchanged: float = 0.0
    avg_skill_levels: dict[str, float] = field(default_factory=dict)
    work_parties_formed: int = 0


class MetricsCollector:
    """Collects time-series data every day."""

    def __init__(self) -> None:
        self.snapshots: list[DailySnapshot] = []
        self._daily_births: int = 0
        self._daily_deaths: int = 0
        self._daily_marriages: int = 0
        self._daily_trades: int = 0
        self._daily_trade_items: float = 0.0
        self._daily_work_parties: int = 0

    def record_birth(self) -> None:
        self._daily_births += 1

    def record_death(self) -> None:
        self._daily_deaths += 1

    def record_marriage(self) -> None:
        self._daily_marriages += 1

    def record_trade(self, items_exchanged: float = 0.0) -> None:
        self._daily_trades += 1
        self._daily_trade_items += items_exchanged

    def record_work_party(self) -> None:
        self._daily_work_parties += 1

    def collect_daily(
        self,
        day: int,
        villagers: list["Villager"],  # noqa: F821
        family_manager: "FamilyManager",  # noqa: F821
        resource_manager: "ResourceManager",  # noqa: F821
    ) -> DailySnapshot:
        """Collect all metrics for this day."""
        alive = [v for v in villagers if v.is_alive]
        n = len(alive)

        # Average sentiment
        avg_sentiment = sum(v.current_sentiment for v in alive) / max(1, n)

        # Average hunger satisfaction
        avg_hunger = sum(v.needs.needs["hunger"].satisfaction for v in alive) / max(1, n)

        # Average health
        avg_health = sum(v.health for v in alive) / max(1, n)

        # Total food across all families
        total_food = sum(
            fam.total_food() for fam in family_manager.families.values()
        )
        food_per_capita = total_food / max(1, n)

        # Gini coefficient of family wealth
        family_wealths = [
            fam.inventory.total_weight()
            for fam in family_manager.families.values()
            if fam.member_ids
        ]
        gini = self.gini_coefficient(family_wealths)

        # Activity distribution
        activity_counts: dict[str, int] = {}
        for v in alive:
            act = v.current_activity or "idle"
            activity_counts[act] = activity_counts.get(act, 0) + 1

        # Average wellbeing
        avg_wellbeing = sum(v.needs.overall_wellbeing() for v in alive) / max(1, n)

        # Average skill levels by category
        skill_totals: dict[str, float] = {}
        skill_counts: dict[str, int] = {}
        for v in alive:
            for skill_name, xp in v.memory.skill_experience.items():
                skill_level = v.memory.skill_level(skill_name, v.traits.intelligence)
                skill_totals[skill_name] = skill_totals.get(skill_name, 0.0) + skill_level
                skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
        avg_skill_levels = {
            name: skill_totals[name] / max(1, skill_counts[name])
            for name in skill_totals
        }

        snapshot = DailySnapshot(
            day=day,
            population=n,
            births=self._daily_births,
            deaths=self._daily_deaths,
            avg_sentiment=avg_sentiment,
            avg_hunger=avg_hunger,
            avg_health=avg_health,
            total_food=total_food,
            food_per_capita=food_per_capita,
            gini=gini,
            activity_counts=activity_counts,
            avg_wellbeing=avg_wellbeing,
            marriage_count=self._daily_marriages,
            trade_count=self._daily_trades,
            trade_items_exchanged=self._daily_trade_items,
            avg_skill_levels=avg_skill_levels,
            work_parties_formed=self._daily_work_parties,
        )
        self.snapshots.append(snapshot)

        # Reset daily counters
        self._daily_births = 0
        self._daily_deaths = 0
        self._daily_marriages = 0
        self._daily_trades = 0
        self._daily_trade_items = 0.0
        self._daily_work_parties = 0

        return snapshot

    @staticmethod
    def gini_coefficient(values: list[float]) -> float:
        """Calculate the Gini coefficient of a list of values."""
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = 0.0
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        for i, val in enumerate(sorted_vals):
            cumulative += val
        # Using the formula: G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n

    def export_csv(self, filepath: str) -> None:
        """Export all snapshots to CSV."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "day", "population", "births", "deaths", "avg_sentiment",
                "avg_hunger", "avg_health", "total_food", "food_per_capita",
                "gini", "avg_wellbeing", "marriages", "trades",
                "trade_items", "work_parties",
            ])
            for s in self.snapshots:
                writer.writerow([
                    s.day, s.population, s.births, s.deaths,
                    f"{s.avg_sentiment:.2f}", f"{s.avg_hunger:.3f}",
                    f"{s.avg_health:.1f}", f"{s.total_food:.1f}",
                    f"{s.food_per_capita:.2f}", f"{s.gini:.3f}",
                    f"{s.avg_wellbeing:.3f}", s.marriage_count,
                    s.trade_count, f"{s.trade_items_exchanged:.1f}",
                    s.work_parties_formed,
                ])

    def summary_report(self, start_day: int = 0, end_day: Optional[int] = None) -> str:
        """Generate a human-readable summary of the simulation period."""
        relevant = [
            s for s in self.snapshots
            if s.day >= start_day and (end_day is None or s.day <= end_day)
        ]
        if not relevant:
            return "No data available for the specified period."

        first = relevant[0]
        last = relevant[-1]
        total_births = sum(s.births for s in relevant)
        total_deaths = sum(s.deaths for s in relevant)
        total_marriages = sum(s.marriage_count for s in relevant)
        total_trades = sum(s.trade_count for s in relevant)
        total_trade_items = sum(s.trade_items_exchanged for s in relevant)

        lines = [
            f"=== Simulation Summary: Day {first.day} to Day {last.day} ===",
            f"Duration: {last.day - first.day + 1} days ({(last.day - first.day + 1) / 360:.1f} years)",
            f"",
            f"Population: {first.population} -> {last.population}",
            f"  Total births: {total_births}",
            f"  Total deaths: {total_deaths}",
            f"  Total marriages: {total_marriages}",
            f"",
            f"Economy:",
            f"  Total trades: {total_trades}",
            f"  Total items exchanged: {total_trade_items:.0f}",
            f"  Avg trades/day: {total_trades / max(1, len(relevant)):.1f}",
            f"",
            f"Final Metrics:",
            f"  Avg sentiment: {last.avg_sentiment:.1f}/100",
            f"  Avg hunger satisfaction: {last.avg_hunger:.1%}",
            f"  Avg health: {last.avg_health:.1f}/100",
            f"  Food per capita: {last.food_per_capita:.2f}",
            f"  Wealth inequality (Gini): {last.gini:.3f}",
            f"  Avg wellbeing: {last.avg_wellbeing:.1%}",
        ]

        # Activity distribution
        if last.activity_counts:
            lines.append(f"")
            lines.append(f"Activity Distribution (final day):")
            total_acts = sum(last.activity_counts.values())
            for act, count in sorted(last.activity_counts.items(), key=lambda x: -x[1]):
                pct = count / max(1, total_acts) * 100
                lines.append(f"  {act}: {count} ({pct:.0f}%)")

        # Skill levels
        if last.avg_skill_levels:
            lines.append(f"")
            lines.append(f"Average Skill Levels (final day):")
            for skill, level in sorted(last.avg_skill_levels.items(), key=lambda x: -x[1]):
                if level > 0.5:
                    lines.append(f"  {skill}: {level:.1f}")

        return "\n".join(lines)

"""Real-time matplotlib dashboard for simulation visualization."""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend
import matplotlib.pyplot as plt
import numpy as np

from village_sim.core.config import DASHBOARD_UPDATE_INTERVAL


class Dashboard:
    """Real-time dashboard with 8 subplots updating during simulation."""

    def __init__(self) -> None:
        self._initialized = False
        self._fig = None
        self._axes = None
        self._update_counter = 0

    def initialize(self) -> None:
        """Set up the matplotlib figure and subplots."""
        plt.ion()
        self._fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        self._fig.suptitle("Village Simulation Dashboard", fontsize=14)
        self._axes = {
            "population": axes[0, 0],
            "food": axes[0, 1],
            "sentiment": axes[0, 2],
            "trade": axes[0, 3],
            "gini": axes[1, 0],
            "activities": axes[1, 1],
            "health": axes[1, 2],
            "skills": axes[1, 3],
        }
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        self._axes["population"].set_title("Population")
        self._axes["food"].set_title("Food per Capita")
        self._axes["sentiment"].set_title("Average Sentiment")
        self._axes["trade"].set_title("Trade Volume")
        self._axes["gini"].set_title("Wealth Inequality (Gini)")
        self._axes["activities"].set_title("Activity Distribution")
        self._axes["health"].set_title("Health & Wellbeing")
        self._axes["skills"].set_title("Skill Development")

        plt.tight_layout()
        self._initialized = True
        plt.pause(0.01)

    def update(self, day: int, metrics: "MetricsCollector") -> None:  # noqa: F821
        """Update the dashboard with latest metrics."""
        self._update_counter += 1
        if self._update_counter % DASHBOARD_UPDATE_INTERVAL != 0:
            return

        if not self._initialized:
            self.initialize()

        snapshots = metrics.snapshots
        if not snapshots:
            return

        days = [s.day for s in snapshots]

        # Population
        ax = self._axes["population"]
        ax.clear()
        ax.set_title("Population")
        ax.plot(days, [s.population for s in snapshots], "b-", linewidth=1.5)
        ax.grid(True, alpha=0.3)

        # Food per capita
        ax = self._axes["food"]
        ax.clear()
        ax.set_title("Food per Capita")
        ax.plot(days, [s.food_per_capita for s in snapshots], "g-", linewidth=1.5)
        ax.axhline(y=2.0, color="r", linestyle="--", alpha=0.5, label="Min need")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Sentiment
        ax = self._axes["sentiment"]
        ax.clear()
        ax.set_title("Average Sentiment")
        ax.plot(days, [s.avg_sentiment for s in snapshots], "m-", linewidth=1.5)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Trade volume
        ax = self._axes["trade"]
        ax.clear()
        ax.set_title("Trade Volume")
        ax.plot(days, [s.trade_count for s in snapshots], "c-", linewidth=1.5, label="Trades/day")
        ax.set_ylabel("Trades")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Gini
        ax = self._axes["gini"]
        ax.clear()
        ax.set_title("Wealth Inequality (Gini)")
        ax.plot(days, [s.gini for s in snapshots], "r-", linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Activity distribution (pie chart of latest day)
        ax = self._axes["activities"]
        ax.clear()
        ax.set_title("Activity Distribution")
        latest = snapshots[-1]
        if latest.activity_counts:
            sorted_acts = sorted(latest.activity_counts.items(), key=lambda x: -x[1])
            top = sorted_acts[:8]
            if len(sorted_acts) > 8:
                other_count = sum(c for _, c in sorted_acts[8:])
                top.append(("other", other_count))
            labels = [a for a, _ in top]
            sizes = [c for _, c in top]
            ax.pie(sizes, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 7})

        # Health & Wellbeing
        ax = self._axes["health"]
        ax.clear()
        ax.set_title("Health & Wellbeing")
        ax.plot(days, [s.avg_health for s in snapshots], "b-", label="Health", linewidth=1.5)
        ax.plot(
            days,
            [s.avg_wellbeing * 100 for s in snapshots],
            "g-", label="Wellbeing", linewidth=1.5,
        )
        ax.plot(
            days,
            [s.avg_hunger * 100 for s in snapshots],
            "orange", label="Hunger Sat.", linewidth=1,
        )
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Skill development
        ax = self._axes["skills"]
        ax.clear()
        ax.set_title("Skill Development")
        # Collect skill data across all snapshots
        all_skills: set[str] = set()
        for s in snapshots:
            all_skills.update(s.avg_skill_levels.keys())

        top_skills = sorted(
            all_skills,
            key=lambda sk: snapshots[-1].avg_skill_levels.get(sk, 0),
            reverse=True,
        )[:6]  # top 6 skills

        for skill_name in top_skills:
            values = [s.avg_skill_levels.get(skill_name, 0) for s in snapshots]
            if max(values) > 0.5:
                ax.plot(days, values, linewidth=1, label=skill_name)

        if top_skills:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        self._fig.suptitle(f"Village Simulation — Day {day}", fontsize=14)
        plt.tight_layout()
        plt.pause(0.01)

    def save(self, filepath: str) -> None:
        """Save the current dashboard as an image."""
        if self._fig:
            self._fig.savefig(filepath, dpi=150, bbox_inches="tight")

    def close(self) -> None:
        """Close the dashboard."""
        if self._fig:
            plt.close(self._fig)

    # ------------------------------------------------------------------
    # Post-hoc static plots
    # ------------------------------------------------------------------

    @staticmethod
    def comprehensive_report(metrics: "MetricsCollector", output_dir: str) -> None:  # noqa: F821
        """Generate all plots and save to output directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        snapshots = metrics.snapshots
        if not snapshots:
            return

        days = [s.day for s in snapshots]

        # Population over time
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, [s.population for s in snapshots])
        ax.set_title("Population Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "population.png"), dpi=150)
        plt.close(fig)

        # Food security
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, [s.food_per_capita for s in snapshots])
        ax.set_title("Food per Capita Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Food Units")
        ax.axhline(y=2.0, color="r", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "food_security.png"), dpi=150)
        plt.close(fig)

        # Sentiment
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, [s.avg_sentiment for s in snapshots])
        ax.set_title("Average Sentiment Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Sentiment (0-100)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "sentiment.png"), dpi=150)
        plt.close(fig)

        # Inequality
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, [s.gini for s in snapshots])
        ax.set_title("Wealth Inequality (Gini Coefficient)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Gini")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "inequality.png"), dpi=150)
        plt.close(fig)

        # Trade volume
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, [s.trade_count for s in snapshots], "c-")
        ax.set_title("Trade Volume Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Trades per Day")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "trade_volume.png"), dpi=150)
        plt.close(fig)

        # Skill development
        fig, ax = plt.subplots(figsize=(10, 5))
        all_skills: set[str] = set()
        for s in snapshots:
            all_skills.update(s.avg_skill_levels.keys())
        top_skills = sorted(
            all_skills,
            key=lambda sk: snapshots[-1].avg_skill_levels.get(sk, 0),
            reverse=True,
        )[:8]
        for skill_name in top_skills:
            values = [s.avg_skill_levels.get(skill_name, 0) for s in snapshots]
            if max(values) > 0.5:
                ax.plot(days, values, linewidth=1.5, label=skill_name)
        ax.set_title("Skill Development Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Average Skill Level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "skill_development.png"), dpi=150)
        plt.close(fig)

        print(f"Reports saved to {output_dir}/")

"""Live-updating per-villager dashboard — track one villager's life in real time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from village_sim.core.config import TRACKER_UPDATE_INTERVAL
from village_sim.viz.grid_view import ACTIVITY_COLORS, NEED_COLORS

if TYPE_CHECKING:
    from village_sim.simulation.engine import SimulationEngine
    from village_sim.agents.villager import Villager


# =========================================================================
# Snapshot: lightweight per-day copy of villager state
# =========================================================================

@dataclass
class VillagerDaySnapshot:
    """One day's worth of tracked villager data."""

    day: int
    health: float           # 0-100
    fatigue: float          # 0-1
    sentiment: float        # 0-100
    needs: dict[str, float] = field(default_factory=dict)  # name -> satisfaction 0-1
    activity: str = ""
    is_alive: bool = True


# =========================================================================
# Theme constants
# =========================================================================

_BG = "#1a1a2e"
_PANEL_BG = "#16213e"
_TEXT = "#e0e0e0"
_GRID_ALPHA = 0.2
_GRID_COLOR = "#444466"

_SURVIVAL_NEEDS = ["hunger", "thirst", "rest", "warmth", "health"]
_HIGHER_NEEDS = ["shelter", "safety", "social", "purpose", "comfort"]


# =========================================================================
# VillagerTracker
# =========================================================================

class VillagerTracker:
    """Live-updating 6-panel matplotlib figure tracking a single villager."""

    def __init__(
        self,
        engine: "SimulationEngine",
        villager_id: int,
        update_interval: int = TRACKER_UPDATE_INTERVAL,
    ) -> None:
        # Look up the villager
        villager = engine._villager_map.get(villager_id)
        if villager is None:
            raise ValueError(
                f"Villager ID {villager_id} not found. "
                f"Valid IDs: 0-{max(engine._villager_map.keys())}"
            )

        self._villager: "Villager" = villager
        self._engine = engine
        self._update_interval = max(1, update_interval)
        self._update_counter = 0

        # Data history
        self._history: list[VillagerDaySnapshot] = []
        self._dead_day: Optional[int] = None

        # Figure state
        self._fig = None
        self._axes: dict[str, plt.Axes] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the matplotlib figure and 6 subplots."""
        plt.ion()

        self._fig, axes_grid = plt.subplots(
            2, 3, figsize=(20, 10),
            facecolor=_BG,
        )

        self._axes = {
            "vitals":         axes_grid[0, 0],
            "survival_needs": axes_grid[0, 1],
            "higher_needs":   axes_grid[0, 2],
            "activity":       axes_grid[1, 0],
            "episodes":       axes_grid[1, 1],
            "biases":         axes_grid[1, 2],
        }

        # Style all axes
        for ax in axes_grid.flat:
            ax.set_facecolor(_PANEL_BG)
            ax.tick_params(colors=_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333355")
            ax.grid(True, alpha=_GRID_ALPHA, color=_GRID_COLOR)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        self._initialized = True
        plt.pause(0.01)

    def update(self, engine: "SimulationEngine") -> None:
        """Call after each engine.tick(). Captures data and redraws periodically."""
        self._capture_snapshot(engine)
        self._update_counter += 1

        if self._update_counter % self._update_interval == 0:
            self._redraw(engine)

    def finalize(self) -> None:
        """Final redraw, then block until the user closes the window."""
        if self._history:
            self._redraw(self._engine)
        plt.ioff()
        plt.show()

    # ------------------------------------------------------------------
    # Data capture
    # ------------------------------------------------------------------

    def _capture_snapshot(self, engine: "SimulationEngine") -> None:
        """Read the villager's current state and append to history."""
        v = self._villager
        snap = VillagerDaySnapshot(
            day=engine.clock.day,
            health=v.health,
            fatigue=v.fatigue,
            sentiment=v.current_sentiment,
            needs={name: need.satisfaction for name, need in v.needs.needs.items()},
            activity=v.current_activity or "rest",
            is_alive=v.is_alive,
        )
        self._history.append(snap)

        if not v.is_alive and self._dead_day is None:
            self._dead_day = engine.clock.day

    # ------------------------------------------------------------------
    # Redraw all 6 panels
    # ------------------------------------------------------------------

    def _redraw(self, engine: "SimulationEngine") -> None:
        if not self._initialized:
            self.initialize()

        if not self._history:
            return

        days = [s.day for s in self._history]
        v = self._villager

        self._draw_vitals(days)
        self._draw_survival_needs(days)
        self._draw_higher_needs(days)
        self._draw_activity_timeline(days)
        self._draw_episodes()
        self._draw_biases()

        # Suptitle header
        age_y = v.age_days // 360
        status = "ALIVE" if v.is_alive else f"DIED Day {self._dead_day}"
        act = self._history[-1].activity if self._history else ""
        self._fig.suptitle(
            f"Tracking: {v.name} (ID {v.id})  |  {v.sex.title()} Age {age_y}y  |  "
            f"Family {v.family_id}  |  {status}  |  Day {engine.clock.day}  |  "
            f"Activity: {act.replace('_', ' ').title()}",
            fontsize=12, fontweight="bold", color=_TEXT,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Panel 1: Vital Signs
    # ------------------------------------------------------------------

    def _draw_vitals(self, days: list[int]) -> None:
        ax = self._axes["vitals"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        health = [s.health for s in self._history]
        sentiment = [s.sentiment for s in self._history]
        fatigue_pct = [s.fatigue * 100 for s in self._history]

        ax.plot(days, health, color="#4488ff", linewidth=2, label="Health")
        ax.plot(days, sentiment, color="#cc44cc", linewidth=1.5, label="Sentiment")
        ax.plot(days, fatigue_pct, color="#ff4444", linewidth=1.5, alpha=0.8, label="Fatigue x100")

        if self._dead_day is not None:
            ax.axvline(self._dead_day, color="white", linestyle="--", linewidth=2, alpha=0.7, label="Death")

        ax.set_ylim(-2, 102)
        ax.set_title("Vital Signs", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", color=_TEXT, fontsize=8)
        ax.legend(fontsize=7, loc="lower left", facecolor=_PANEL_BG, edgecolor="#555577",
                  labelcolor=_TEXT)
        ax.grid(True, alpha=_GRID_ALPHA, color=_GRID_COLOR)
        ax.tick_params(colors=_TEXT, labelsize=8)

    # ------------------------------------------------------------------
    # Panel 2: Survival Needs
    # ------------------------------------------------------------------

    def _draw_survival_needs(self, days: list[int]) -> None:
        ax = self._axes["survival_needs"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        for need_name in _SURVIVAL_NEEDS:
            vals = [s.needs.get(need_name, 0) * 100 for s in self._history]
            color = NEED_COLORS.get(need_name, "#aaaaaa")
            ax.plot(days, vals, color=color, linewidth=1.5,
                    label=need_name.title())

        ax.axhline(20, color="#ff4444", linestyle=":", alpha=0.5, linewidth=1)

        if self._dead_day is not None:
            ax.axvline(self._dead_day, color="white", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_ylim(-2, 102)
        ax.set_title("Survival Needs", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", color=_TEXT, fontsize=8)
        ax.legend(fontsize=7, loc="lower left", facecolor=_PANEL_BG, edgecolor="#555577",
                  labelcolor=_TEXT)
        ax.grid(True, alpha=_GRID_ALPHA, color=_GRID_COLOR)
        ax.tick_params(colors=_TEXT, labelsize=8)

    # ------------------------------------------------------------------
    # Panel 3: Higher Needs
    # ------------------------------------------------------------------

    def _draw_higher_needs(self, days: list[int]) -> None:
        ax = self._axes["higher_needs"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        for need_name in _HIGHER_NEEDS:
            vals = [s.needs.get(need_name, 0) * 100 for s in self._history]
            color = NEED_COLORS.get(need_name, "#aaaaaa")
            ax.plot(days, vals, color=color, linewidth=1.5,
                    label=need_name.title())

        if self._dead_day is not None:
            ax.axvline(self._dead_day, color="white", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_ylim(-2, 102)
        ax.set_title("Higher Needs", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", color=_TEXT, fontsize=8)
        ax.legend(fontsize=7, loc="lower left", facecolor=_PANEL_BG, edgecolor="#555577",
                  labelcolor=_TEXT)
        ax.grid(True, alpha=_GRID_ALPHA, color=_GRID_COLOR)
        ax.tick_params(colors=_TEXT, labelsize=8)

    # ------------------------------------------------------------------
    # Panel 4: Activity Timeline
    # ------------------------------------------------------------------

    def _draw_activity_timeline(self, days: list[int]) -> None:
        ax = self._axes["activity"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        if not self._history:
            return

        # Build contiguous segments for broken_barh
        segments: list[tuple[float, float, str]] = []  # (start, width, color)
        seen_activities: dict[str, str] = {}  # activity -> color (for legend)

        for snap in self._history:
            color = ACTIVITY_COLORS.get(snap.activity, "#888888")
            segments.append((snap.day - 0.5, 1.0, color))
            if snap.activity not in seen_activities:
                seen_activities[snap.activity] = color

        # Draw each segment
        for start, width, color in segments:
            ax.barh(0, width, left=start, height=0.8, color=color, edgecolor="none")

        # Compact legend (up to 10 activities)
        legend_items = list(seen_activities.items())[:10]
        patches = [
            mpatches.Patch(
                color=color,
                label=act.replace("_", " ").title()[:14],
            )
            for act, color in legend_items
        ]
        if patches:
            ax.legend(
                handles=patches, fontsize=6, ncol=min(5, len(patches)),
                loc="upper center", bbox_to_anchor=(0.5, -0.12),
                facecolor=_PANEL_BG, edgecolor="#555577", labelcolor=_TEXT,
            )

        ax.set_yticks([])
        ax.set_title("Activity Timeline", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", color=_TEXT, fontsize=8)
        ax.tick_params(colors=_TEXT, labelsize=8)

        # Show current activity as right-aligned text
        latest = self._history[-1]
        ax.text(
            0.98, 0.85,
            f"Now: {latest.activity.replace('_', ' ').title()}",
            transform=ax.transAxes, fontsize=9, color=_TEXT,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG, edgecolor="#555577", alpha=0.8),
        )

    # ------------------------------------------------------------------
    # Panel 5: Life Events (episodes)
    # ------------------------------------------------------------------

    def _draw_episodes(self) -> None:
        ax = self._axes["episodes"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        episodes = list(self._villager.memory.episodes)

        if not episodes:
            ax.text(
                0.5, 0.5, "No episodes yet",
                transform=ax.transAxes, fontsize=11, color="#888888",
                ha="center", va="center",
            )
            ax.set_title("Life Events", color=_TEXT, fontsize=10, fontweight="bold")
            ax.tick_params(colors=_TEXT, labelsize=8)
            return

        # Draw stems: vlines + scatter
        for ep in episodes:
            color = "#44cc44" if ep.emotional_impact >= 0 else "#cc4444"
            ax.vlines(ep.day, 0, ep.emotional_impact, colors=color, linewidth=1.5, alpha=0.8)
            ax.scatter(ep.day, ep.emotional_impact, color=color, s=18, zorder=5, edgecolors="none")

        ax.axhline(0, color="#666666", linewidth=0.5)

        # Annotate last 3 episodes
        recent = episodes[-3:] if len(episodes) >= 3 else episodes
        for ep in recent:
            va = "bottom" if ep.emotional_impact >= 0 else "top"
            offset = 0.03 if ep.emotional_impact >= 0 else -0.03
            ax.annotate(
                ep.category.replace("_", " ")[:14],
                (ep.day, ep.emotional_impact + offset),
                fontsize=6, color=_TEXT, rotation=25,
                ha="left", va=va, alpha=0.9,
            )

        if self._dead_day is not None:
            ax.axvline(self._dead_day, color="white", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_ylim(-1.1, 1.1)
        ax.set_title("Life Events", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", color=_TEXT, fontsize=8)
        ax.set_ylabel("Impact", color=_TEXT, fontsize=8)
        ax.grid(True, alpha=_GRID_ALPHA, color=_GRID_COLOR)
        ax.tick_params(colors=_TEXT, labelsize=8)

    # ------------------------------------------------------------------
    # Panel 6: Activity Biases
    # ------------------------------------------------------------------

    def _draw_biases(self) -> None:
        ax = self._axes["biases"]
        ax.clear()
        ax.set_facecolor(_PANEL_BG)

        biases = dict(self._villager.memory.activity_biases)

        if not biases:
            ax.text(
                0.5, 0.5, "No biases yet",
                transform=ax.transAxes, fontsize=11, color="#888888",
                ha="center", va="center",
            )
            ax.set_title("Learned Activity Biases", color=_TEXT, fontsize=10, fontweight="bold")
            ax.tick_params(colors=_TEXT, labelsize=8)
            return

        # Sort by value (most negative first)
        sorted_biases = sorted(biases.items(), key=lambda x: x[1])
        names = [name.replace("_", " ").title()[:16] for name, _ in sorted_biases]
        values = [val for _, val in sorted_biases]
        colors = ["#44cc44" if v >= 0 else "#cc4444" for v in values]

        bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.6)

        # Value labels
        for bar, val in zip(bars, values):
            x_pos = val + (0.01 if val >= 0 else -0.01)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", fontsize=7, color=_TEXT, ha=ha, va="center")

        ax.axvline(0, color="#666666", linewidth=0.5)
        ax.set_xlim(-0.35, 0.35)
        ax.set_title("Learned Activity Biases", color=_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Bias (avoid ← → attract)", color=_TEXT, fontsize=8)
        ax.tick_params(colors=_TEXT, labelsize=8, axis="y")
        ax.tick_params(colors=_TEXT, labelsize=7, axis="x")

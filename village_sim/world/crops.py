"""Crop management system — CropPlot lifecycle and CropManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from village_sim.core.config import CROP_FAILURE_FROST_THRESHOLD, CROP_GROWTH_DAYS


@dataclass
class CropPlot:
    """A planted crop at a specific map position."""

    position: tuple[int, int]
    family_id: int
    planted_day: int
    crop_stage: str = "planted"  # planted -> growing -> mature -> harvestable -> failed
    quality: float = 0.5
    expected_yield: float = 10.0
    days_growing: int = 0
    times_tended: int = 0

    def daily_growth(self, climate: "Climate", was_tended_today: bool) -> None:  # noqa: F821
        """Advance crop growth by one day."""
        if self.crop_stage == "failed":
            return

        self.days_growing += 1

        # Frost kills crops
        if climate.temperature < CROP_FAILURE_FROST_THRESHOLD and self.crop_stage != "harvestable":
            self.crop_stage = "failed"
            return

        # Tending improves quality
        if was_tended_today:
            self.times_tended += 1
            self.quality = min(1.0, self.quality + 0.02)

        # Weather affects quality
        growth_mod = climate.crop_growth_modifier()
        self.quality = max(0.0, min(1.0, self.quality + (growth_mod - 0.8) * 0.01))

        # Stage transitions based on days
        growth_frac = self.days_growing / CROP_GROWTH_DAYS
        if growth_frac < 0.3:
            self.crop_stage = "planted"
        elif growth_frac < 0.7:
            self.crop_stage = "growing"
        elif growth_frac < 1.0:
            self.crop_stage = "mature"
        else:
            self.crop_stage = "harvestable"

        # Update expected yield based on quality
        self.expected_yield = 10.0 * self.quality * (1 + 0.1 * self.times_tended)


class CropManager:
    """Manages all crop plots in the world."""

    def __init__(self) -> None:
        self.plots: list[CropPlot] = []

    def plant(
        self, position: tuple[int, int], family_id: int, current_day: int
    ) -> CropPlot:
        """Create a new crop plot."""
        plot = CropPlot(
            position=position,
            family_id=family_id,
            planted_day=current_day,
        )
        self.plots.append(plot)
        return plot

    def daily_update(self, climate: "Climate", tended_positions: set[tuple[int, int]]) -> None:  # noqa: F821
        """Update all crop plots for the day."""
        for plot in self.plots:
            was_tended = plot.position in tended_positions
            plot.daily_growth(climate, was_tended)

    def get_harvestable(self, family_id: Optional[int] = None) -> list[CropPlot]:
        """Get all harvestable crop plots, optionally filtered by family."""
        result = [p for p in self.plots if p.crop_stage == "harvestable"]
        if family_id is not None:
            result = [p for p in result if p.family_id == family_id]
        return result

    def get_family_plots(self, family_id: int) -> list[CropPlot]:
        """Get all active plots for a family."""
        return [
            p for p in self.plots
            if p.family_id == family_id and p.crop_stage != "failed"
        ]

    def remove_harvested(self, plot: CropPlot) -> None:
        """Remove a harvested or failed crop plot."""
        if plot in self.plots:
            self.plots.remove(plot)

    def cleanup_failed(self) -> list[CropPlot]:
        """Remove and return all failed crop plots."""
        failed = [p for p in self.plots if p.crop_stage == "failed"]
        self.plots = [p for p in self.plots if p.crop_stage != "failed"]
        return failed

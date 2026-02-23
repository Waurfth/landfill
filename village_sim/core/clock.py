"""Time system for the simulation: ticks, days, seasons, years."""

from village_sim.core.config import DAYS_PER_SEASON, DAYS_PER_YEAR, SEASONS, DAYLIGHT_HOURS


class SimClock:
    """Manages simulation time."""

    def __init__(self) -> None:
        self.day: int = 0

    @property
    def season(self) -> str:
        season_index = (self.day_of_year // DAYS_PER_SEASON) % len(SEASONS)
        return SEASONS[season_index]

    @property
    def year(self) -> int:
        return self.day // DAYS_PER_YEAR

    @property
    def day_of_year(self) -> int:
        return self.day % DAYS_PER_YEAR

    @property
    def day_of_season(self) -> int:
        return self.day_of_year % DAYS_PER_SEASON

    def advance(self) -> None:
        """Advance the clock by one day."""
        self.day += 1

    def is_planting_season(self) -> bool:
        return self.season == "spring"

    def is_harvest_season(self) -> bool:
        return self.season == "autumn"

    def daylight_hours(self) -> float:
        """Hours of daylight, varies by season with smooth interpolation."""
        season = self.season
        day_frac = self.day_of_season / DAYS_PER_SEASON

        current_hours = DAYLIGHT_HOURS[season]
        next_season = SEASONS[(SEASONS.index(season) + 1) % len(SEASONS)]
        next_hours = DAYLIGHT_HOURS[next_season]

        # Linear interpolation within the season toward the next
        return current_hours + (next_hours - current_hours) * day_frac

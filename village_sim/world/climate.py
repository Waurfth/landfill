"""Seasons, weather events, and terrain/work modifiers."""

from __future__ import annotations

from numpy.random import Generator


# Base temperature ranges by season (abstract 0-100 scale)
_TEMP_RANGES: dict[str, tuple[float, float]] = {
    "spring": (40.0, 60.0),
    "summer": (60.0, 85.0),
    "autumn": (35.0, 55.0),
    "winter": (10.0, 35.0),
}

# Weather probabilities by season: weather_type -> probability
_WEATHER_PROBS: dict[str, dict[str, float]] = {
    "spring": {"clear": 0.50, "rain": 0.30, "storm": 0.05, "fog": 0.10, "heat_wave": 0.02, "snow": 0.03},
    "summer": {"clear": 0.55, "rain": 0.20, "storm": 0.08, "fog": 0.05, "heat_wave": 0.12, "snow": 0.00},
    "autumn": {"clear": 0.35, "rain": 0.30, "storm": 0.15, "fog": 0.15, "heat_wave": 0.01, "snow": 0.04},
    "winter": {"clear": 0.30, "rain": 0.15, "storm": 0.10, "fog": 0.10, "heat_wave": 0.00, "snow": 0.35},
}


class Climate:
    """Daily weather generation and environmental modifiers."""

    def __init__(self, rng: Generator) -> None:
        self._rng = rng
        self.current_weather: str = "clear"
        self.temperature: float = 50.0
        self.consecutive_dry_days: int = 0

    def advance_day(self, season: str, day_of_season: int) -> None:
        """Generate weather for the new day."""
        # Temperature: base range with daily noise
        lo, hi = _TEMP_RANGES.get(season, (40.0, 60.0))
        base_temp = lo + (hi - lo) * (day_of_season / 90.0 * 0.3 + 0.35)
        self.temperature = float(
            max(0, min(100, base_temp + self._rng.normal(0, 5)))
        )

        # Weather: weighted random choice
        probs = _WEATHER_PROBS.get(season, _WEATHER_PROBS["spring"])
        weather_types = list(probs.keys())
        weights = [probs[w] for w in weather_types]
        total = sum(weights)
        weights = [w / total for w in weights]
        self.current_weather = self._rng.choice(weather_types, p=weights)

        # Track dry days
        if self.current_weather in ("rain", "storm", "snow"):
            self.consecutive_dry_days = 0
        else:
            self.consecutive_dry_days += 1

    def outdoor_work_modifier(self) -> float:
        """Modifier for outdoor activity productivity."""
        modifiers = {
            "clear": 1.0,
            "rain": 0.7,
            "storm": 0.3,
            "fog": 0.85,
            "heat_wave": 0.6,
            "snow": 0.5,
        }
        return modifiers.get(self.current_weather, 1.0)

    def warmth_need_modifier(self) -> float:
        """Higher when cold, lower when warm. Scales warmth need decay."""
        if self.temperature < 20:
            return 2.5
        if self.temperature < 35:
            return 1.5
        if self.temperature < 50:
            return 1.0
        if self.temperature < 70:
            return 0.5
        return 0.3

    def crop_growth_modifier(self) -> float:
        """How favorable is today's weather for crop growth."""
        base = 1.0
        if self.current_weather == "rain":
            base = 1.2
        elif self.current_weather == "storm":
            base = 0.6
        elif self.current_weather == "heat_wave":
            base = 0.5
        elif self.current_weather == "snow":
            base = 0.0

        # Temperature effect
        if self.temperature < 15:
            base *= 0.1  # near frost
        elif self.temperature > 80:
            base *= 0.6  # too hot

        return base

    def shelter_damage_modifier(self) -> float:
        """Extra damage to shelters from weather."""
        if self.current_weather == "storm":
            return 5.0
        if self.current_weather == "snow":
            return 2.0
        return 1.0

    def terrain_weather_modifier(self, terrain_type: str) -> float:
        """Extra movement cost modifier from weather on specific terrain."""
        if self.current_weather in ("rain", "storm"):
            if terrain_type == "swamp":
                return 1.5
            if terrain_type in ("hills", "rocky"):
                return 1.2
        if self.current_weather == "snow":
            return 1.3
        return 1.0

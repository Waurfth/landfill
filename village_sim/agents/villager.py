"""Core agent class: traits, state, aging, lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

from village_sim.agents.memory import Memory
from village_sim.agents.needs import NeedSystem
from village_sim.agents.personality import PersonalityTraits, generate_personality, inherit_traits
from village_sim.core.config import (
    CHILD_MATURITY_AGE,
    DAYS_PER_YEAR,
    ELDER_DECLINE_AGE,
    FERTILITY_AGE_RANGE,
    INITIAL_POPULATION,
    MAX_AGE,
    POST_BIRTH_RECOVERY_DAYS,
    PREGNANCY_DURATION_DAYS,
    SENTIMENT_DECAY_TOWARD_BASELINE,
    VILLAGE_CENTER,
)


# Simple name lists
_MALE_NAMES: list[str] = [
    "Aldric", "Bran", "Cedric", "Darian", "Edwin", "Finn", "Gareth", "Hadric",
    "Ivor", "Jasper", "Kael", "Leoric", "Magnus", "Nolan", "Oswin", "Perin",
    "Quentin", "Rodric", "Silas", "Theron", "Ulric", "Voss", "Wren", "Yorick",
    "Zander", "Alden", "Beric", "Corwin", "Dorian", "Elric", "Faron", "Gideon",
    "Hugo", "Ivan", "Jorin", "Keldan", "Liam", "Merric", "Niall", "Orin",
]

_FEMALE_NAMES: list[str] = [
    "Adara", "Brynn", "Celia", "Dara", "Elara", "Fiona", "Gwen", "Helena",
    "Iris", "Jessa", "Kira", "Lyra", "Maren", "Nessa", "Olwen", "Petra",
    "Quinn", "Rhea", "Seren", "Thea", "Una", "Vera", "Willa", "Yara",
    "Zara", "Anya", "Blythe", "Clara", "Della", "Eva", "Freya", "Hana",
    "Isla", "Juno", "Keira", "Luna", "Mira", "Nell", "Opal", "Rowan",
]


def _random_name(sex: str, rng: Generator) -> str:
    names = _MALE_NAMES if sex == "male" else _FEMALE_NAMES
    return rng.choice(names)


class Villager:
    """A single village agent with personality, needs, memory, and lifecycle."""

    def __init__(
        self,
        villager_id: int,
        name: str,
        sex: str,
        age_days: int,
        traits: PersonalityTraits,
        birth_day: int = 0,
    ) -> None:
        self.id = villager_id
        self.name = name
        self.sex = sex
        self.age_days = age_days
        self.birth_day = birth_day

        # Components
        self.traits = traits
        self.needs = NeedSystem()
        self.memory = Memory()

        # Physical state
        self.health: float = 100.0
        self.fatigue: float = 0.0
        self.is_pregnant: bool = False
        self.pregnancy_days: int = 0
        self.recovery_days: int = 0  # post-birth recovery
        self.is_alive: bool = True

        # Current state
        self.current_activity: str = ""
        self.current_position: tuple[int, int] = VILLAGE_CENTER
        self.home_position: tuple[int, int] = VILLAGE_CENTER

        # Social
        self.family_id: int = -1
        self.spouse_id: Optional[int] = None
        self.parent_ids: list[int] = []
        self.children_ids: list[int] = []

        # Sentiment: 0 (despair) to 100 (euphoric)
        self.current_sentiment: float = traits.baseline_optimism

        # Inventory assigned externally by engine
        self.personal_inventory: Optional["Inventory"] = None  # noqa: F821

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def age_years(self) -> int:
        return self.age_days // DAYS_PER_YEAR

    @property
    def is_child(self) -> bool:
        return self.age_years < CHILD_MATURITY_AGE

    @property
    def is_elder(self) -> bool:
        return self.age_years > ELDER_DECLINE_AGE

    @property
    def is_fertile(self) -> bool:
        if self.sex != "female":
            return False
        lo, hi = FERTILITY_AGE_RANGE
        return lo <= self.age_years <= hi and not self.is_pregnant

    @property
    def effective_strength(self) -> float:
        return self.traits.strength * self._age_physical_modifier() * self._health_modifier() * self._fatigue_modifier()

    @property
    def effective_endurance(self) -> float:
        return self.traits.endurance * self._age_physical_modifier() * self._health_modifier() * self._fatigue_modifier()

    @property
    def effective_dexterity(self) -> float:
        return self.traits.dexterity * self._age_physical_modifier() * self._health_modifier() * self._fatigue_modifier()

    def get_effective_trait(self, trait_name: str) -> float:
        """Get a trait value modified by age/health/fatigue for physical traits."""
        base = getattr(self.traits, trait_name, 50.0)
        if trait_name in ("strength", "endurance", "dexterity"):
            return base * self._age_physical_modifier() * self._health_modifier() * self._fatigue_modifier()
        if trait_name == "intelligence":
            return base * self._age_mental_modifier()
        return base

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def _age_physical_modifier(self) -> float:
        """Physical trait modifier based on age."""
        age = self.age_years
        if age < CHILD_MATURITY_AGE:
            return max(0.3, age / CHILD_MATURITY_AGE)
        if age <= 20:
            return 0.8 + 0.2 * ((age - CHILD_MATURITY_AGE) / (20 - CHILD_MATURITY_AGE))
        if age <= 40:
            return 1.0
        if age <= ELDER_DECLINE_AGE:
            return 1.0 - 0.15 * ((age - 40) / (ELDER_DECLINE_AGE - 40))
        # Elder
        return max(0.3, 0.85 - 0.02 * (age - ELDER_DECLINE_AGE))

    def _age_mental_modifier(self) -> float:
        """Mental traits are more resilient to aging."""
        age = self.age_years
        if age < 10:
            return max(0.5, age / 10.0)
        if age < 70:
            return 1.0
        return max(0.7, 1.0 - 0.01 * (age - 70))

    def _health_modifier(self) -> float:
        return self.health / 100.0

    def _fatigue_modifier(self) -> float:
        return max(0.3, 1.0 - self.fatigue * 0.5)

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def daily_update(self, day: int, climate: "Climate", rng: Generator) -> None:  # noqa: F821
        """Process one day of aging, sentiment, pregnancy, etc."""
        self.age_days += 1

        # Sentiment drift toward baseline
        baseline = self.traits.baseline_optimism
        diff = baseline - self.current_sentiment
        drift = SENTIMENT_DECAY_TOWARD_BASELINE * diff
        # Recent events influence
        recent_sentiment = self.memory.recall_sentiment()
        self.current_sentiment += drift + recent_sentiment * 0.1
        self.current_sentiment = max(0.0, min(100.0, self.current_sentiment))

        # Elder stat decline
        if self.is_elder and rng.random() < 0.01:
            self.health = max(0, self.health - rng.uniform(0.5, 2.0))

        # Death check (old age or critically low health)
        if self.health <= 0:
            self.die("health_failure")
            return
        if self.age_years >= MAX_AGE:
            # Increasing chance of death past max age
            death_chance = 0.01 * (self.age_years - MAX_AGE + 1)
            if rng.random() < death_chance:
                self.die("old_age")
                return

        # Pregnancy
        if self.is_pregnant:
            self.pregnancy_days += 1

        # Post-birth recovery
        if self.recovery_days > 0:
            self.recovery_days -= 1

        # Fatigue recovery from sleep
        sleep_recovery = 0.6 * (self.health / 100.0)
        self.fatigue = max(0.0, self.fatigue - sleep_recovery)

    def die(self, cause: str = "unknown") -> None:
        """Mark as dead."""
        self.is_alive = False
        self.current_activity = ""

    def give_birth(
        self, villager_id: int, day: int, partner_traits: PersonalityTraits, rng: Generator
    ) -> "Villager":
        """Create a child villager."""
        child_sex = rng.choice(["male", "female"])
        child_name = _random_name(child_sex, rng)
        child_traits = inherit_traits(self.traits, partner_traits, child_sex, rng)

        child = Villager(
            villager_id=villager_id,
            name=child_name,
            sex=child_sex,
            age_days=0,
            traits=child_traits,
            birth_day=day,
        )
        child.parent_ids = [self.id]
        if self.spouse_id is not None:
            child.parent_ids.append(self.spouse_id)
        child.family_id = self.family_id
        child.current_position = self.current_position
        child.home_position = self.home_position

        # Mother recovery
        self.is_pregnant = False
        self.pregnancy_days = 0
        self.recovery_days = POST_BIRTH_RECOVERY_DAYS
        self.children_ids.append(child.id)

        return child


# ------------------------------------------------------------------
# Initial population generation
# ------------------------------------------------------------------

def generate_initial_population(n: int, rng: Generator) -> list[Villager]:
    """Generate the initial village population with families."""
    villagers: list[Villager] = []
    next_id = 0

    # Generate individual villagers
    for _ in range(n):
        sex = rng.choice(["male", "female"])
        name = _random_name(sex, rng)
        age_years = int(np.clip(rng.normal(30, 12), 5, 65))
        age_days = age_years * DAYS_PER_YEAR + rng.integers(0, DAYS_PER_YEAR)
        traits = generate_personality(sex, rng)

        v = Villager(
            villager_id=next_id,
            name=name,
            sex=sex,
            age_days=age_days,
            traits=traits,
        )

        # Partially satisfied needs
        for need in v.needs.needs.values():
            need.satisfaction = rng.uniform(0.5, 1.0)

        villagers.append(v)
        next_id += 1

    # Form married couples from compatible adults
    _assign_marriages(villagers, rng)

    return villagers


def _assign_marriages(villagers: list[Villager], rng: Generator) -> None:
    """Pair compatible adults as married couples."""
    eligible_males = [v for v in villagers if v.sex == "male" and 18 <= v.age_years <= 60]
    eligible_females = [v for v in villagers if v.sex == "female" and 18 <= v.age_years <= 60]
    rng.shuffle(eligible_males)
    rng.shuffle(eligible_females)

    family_id = 0
    paired = min(len(eligible_males), len(eligible_females))
    # Pair ~60% of eligible adults
    pair_count = int(paired * 0.6)

    for i in range(pair_count):
        m = eligible_males[i]
        f = eligible_females[i]
        # Only pair if age difference is reasonable
        if abs(m.age_years - f.age_years) > 15:
            continue

        m.spouse_id = f.id
        f.spouse_id = m.id
        m.family_id = family_id
        f.family_id = family_id

        # Some couples have children
        if m.age_years > 22 and f.age_years > 20:
            num_children = int(rng.integers(0, 4))
            child_villagers = [
                v for v in villagers
                if v.family_id == -1 and v.is_child and v.age_years < min(m.age_years, f.age_years) - 16
            ]
            for child in child_villagers[:num_children]:
                child.family_id = family_id
                child.parent_ids = [m.id, f.id]
                m.children_ids.append(child.id)
                f.children_ids.append(child.id)

        family_id += 1

    # Assign remaining villagers to solo families
    for v in villagers:
        if v.family_id == -1:
            v.family_id = family_id
            family_id += 1

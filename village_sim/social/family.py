"""Family units: food sharing, marriage, household management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from village_sim.core.config import (
    BASE_DAILY_FOOD_NEED,
    CHILD_MATURITY_AGE,
    MARRIAGE_MIN_AFFINITY,
)
from village_sim.economy.inventory import FamilyInventory, food_items


@dataclass
class Family:
    """A family unit — the basic economic and social group."""

    family_id: int
    member_ids: list[int] = field(default_factory=list)
    head_of_household: int = -1
    home_position: tuple[int, int] = (40, 40)
    inventory: FamilyInventory = field(default_factory=lambda: FamilyInventory(0))
    shelter_id: Optional[int] = None
    farm_plots: list = field(default_factory=list)  # CropPlot references

    def __post_init__(self) -> None:
        self.inventory.family_id = self.family_id

    def add_member(self, villager_id: int) -> None:
        if villager_id not in self.member_ids:
            self.member_ids.append(villager_id)

    def remove_member(self, villager_id: int) -> None:
        if villager_id in self.member_ids:
            self.member_ids.remove(villager_id)
        if self.head_of_household == villager_id and self.member_ids:
            self.head_of_household = self.member_ids[0]

    def mouths_to_feed(self, villagers_by_id: dict[int, "Villager"]) -> float:  # noqa: F821
        """How many food-units worth of mouths. Children eat less."""
        total = 0.0
        for vid in self.member_ids:
            v = villagers_by_id.get(vid)
            if v is None or not v.is_alive:
                continue
            if v.is_child:
                total += 0.5
            else:
                total += 1.0
        return total

    def total_food(self) -> float:
        """Total food value in family inventory."""
        return self.inventory.total_food_value()

    def distribute_food(self, villagers_by_id: dict[int, "Villager"]) -> None:  # noqa: F821
        """
        Distribute food from family inventory to hungry members.
        Eat most perishable food first.
        """
        foods = self.inventory.get_all_food()
        if not foods:
            return

        # Calculate total food needed
        for vid in self.member_ids:
            v = villagers_by_id.get(vid)
            if v is None or not v.is_alive:
                continue

            hunger = v.needs.needs["hunger"]
            deficit = 1.0 - hunger.satisfaction
            if deficit <= 0.05:
                continue

            # How much food value needed
            needed = deficit * BASE_DAILY_FOOD_NEED
            consumed = 0.0

            for food_item in foods:
                if food_item.quantity <= 0 or food_item.food_value <= 0:
                    continue
                # How many units to eat
                units_needed = (needed - consumed) / food_item.food_value
                units_to_eat = min(units_needed, food_item.quantity)
                consumed += units_to_eat * food_item.food_value
                food_item.quantity -= units_to_eat

                if consumed >= needed:
                    break

            # Satisfy hunger based on food consumed
            satisfaction_gain = consumed / BASE_DAILY_FOOD_NEED
            v.needs.satisfy("hunger", satisfaction_gain)

        # Clean up empty food stacks
        for item_type in list(self.inventory.items.keys()):
            self.inventory.items[item_type] = [
                s for s in self.inventory.items[item_type] if s.quantity > 0.01
            ]
            if not self.inventory.items[item_type]:
                del self.inventory.items[item_type]

    def daily_needs_check(self, villagers_by_id: dict[int, "Villager"]) -> bool:  # noqa: F821
        """Check if the family is food-secure. Returns True if OK."""
        food = self.total_food()
        mouths = self.mouths_to_feed(villagers_by_id)
        # Food-secure if we have at least 3 days of food
        return food >= mouths * BASE_DAILY_FOOD_NEED * 3


class FamilyManager:
    """Manages all family units."""

    def __init__(self) -> None:
        self.families: dict[int, Family] = {}
        self._next_id: int = 0

    def create_family(
        self, founding_members: list[int], home_position: tuple[int, int] = (40, 40)
    ) -> Family:
        fam = Family(
            family_id=self._next_id,
            member_ids=list(founding_members),
            head_of_household=founding_members[0] if founding_members else -1,
            home_position=home_position,
        )
        self.families[fam.family_id] = fam
        self._next_id += 1
        return fam

    def get_family(self, family_id: int) -> Optional[Family]:
        return self.families.get(family_id)

    def build_from_villagers(self, villagers: list["Villager"]) -> None:  # noqa: F821
        """Build family structures from villager family_id assignments."""
        family_groups: dict[int, list[int]] = {}
        for v in villagers:
            if v.family_id not in family_groups:
                family_groups[v.family_id] = []
            family_groups[v.family_id].append(v.id)

        for fam_id, members in family_groups.items():
            fam = Family(
                family_id=fam_id,
                member_ids=members,
                head_of_household=members[0],
            )
            self.families[fam_id] = fam
            self._next_id = max(self._next_id, fam_id + 1)

    def form_marriage(
        self,
        villager_a: "Villager",  # noqa: F821
        villager_b: "Villager",  # noqa: F821
    ) -> Family:
        """Create or merge family for a married couple."""
        # If both are already in families, merge into A's family
        fam_a = self.families.get(villager_a.family_id)
        fam_b = self.families.get(villager_b.family_id)

        if fam_a is not None and fam_b is not None and fam_a.family_id != fam_b.family_id:
            # Move B into A's family
            fam_b.remove_member(villager_b.id)
            fam_a.add_member(villager_b.id)
            villager_b.family_id = fam_a.family_id
            # If B's family is now empty, remove it
            if not fam_b.member_ids:
                del self.families[fam_b.family_id]
            return fam_a
        elif fam_a is not None:
            fam_a.add_member(villager_b.id)
            villager_b.family_id = fam_a.family_id
            return fam_a
        else:
            return self.create_family([villager_a.id, villager_b.id])

    def split_family(
        self,
        family: Family,
        departing_ids: list[int],
        home_position: tuple[int, int] = (40, 40),
    ) -> Family:
        """Split a family when children grow up and leave."""
        for vid in departing_ids:
            family.remove_member(vid)
        return self.create_family(departing_ids, home_position)

"""Crafting recipes: transforming items with tool requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recipe:
    """A crafting recipe for transforming items."""

    name: str
    inputs: dict[str, float]            # item_type -> quantity consumed
    outputs: dict[str, float]           # item_type -> quantity produced
    tool_requirements: list[str] = field(default_factory=list)  # tool_types needed (not consumed)
    activity: str = "craft_tools"       # which activity category
    skill_requirement: float = 0.0      # minimum skill level to attempt
    quality_from_skill: bool = True     # output quality scales with skill

    def can_craft(
        self,
        inventory: "Inventory",  # noqa: F821
        skill_level: float = 0.0,
    ) -> bool:
        """Check if the villager has materials, tools, and skill to craft this."""
        if skill_level < self.skill_requirement:
            return False
        # Check consumed ingredients
        for item_type, qty in self.inputs.items():
            if not inventory.has(item_type, qty):
                return False
        # Check tool requirements (not consumed)
        for tool_type in self.tool_requirements:
            if not inventory.has_tool_type(tool_type):
                return False
        return True


# =============================================================================
# Recipe definitions
# =============================================================================

RECIPES: dict[str, Recipe] = {
    "stone_axe": Recipe(
        name="stone_axe",
        inputs={"stone": 1, "timber": 1, "plant_fiber": 1},
        outputs={"stone_axe": 1},
        skill_requirement=2,
    ),
    "stone_knife": Recipe(
        name="stone_knife",
        inputs={"stone": 1},
        outputs={"stone_knife": 1},
        skill_requirement=0,
    ),
    "wooden_spear": Recipe(
        name="wooden_spear",
        inputs={"timber": 1},
        outputs={"wooden_spear": 1},
        tool_requirements=["knife"],
        skill_requirement=2,
    ),
    "bow": Recipe(
        name="bow",
        inputs={"timber": 1, "plant_fiber": 2},
        outputs={"bow": 1},
        tool_requirements=["knife"],
        skill_requirement=20,
    ),
    "arrows": Recipe(
        name="arrows",
        inputs={"timber": 0.5, "stone": 0.5},
        outputs={"arrows": 5},
        tool_requirements=["knife"],
        skill_requirement=10,
    ),
    "fishing_rod": Recipe(
        name="fishing_rod",
        inputs={"timber": 1, "plant_fiber": 1},
        outputs={"fishing_rod": 1},
        tool_requirements=["knife"],
        skill_requirement=5,
    ),
    "hoe": Recipe(
        name="hoe",
        inputs={"timber": 1, "stone": 1},
        outputs={"hoe": 1},
        skill_requirement=5,
    ),
    "pickaxe": Recipe(
        name="pickaxe",
        inputs={"timber": 1, "stone": 2},
        outputs={"pickaxe": 1},
        skill_requirement=15,
    ),
    "hammer": Recipe(
        name="hammer",
        inputs={"timber": 1, "stone": 1},
        outputs={"hammer": 1},
        skill_requirement=10,
    ),
    "rope": Recipe(
        name="rope",
        inputs={"plant_fiber": 3},
        outputs={"rope": 1},
        skill_requirement=5,
    ),
    "basket": Recipe(
        name="basket",
        inputs={"plant_fiber": 4},
        outputs={"basket": 1},
        skill_requirement=10,
    ),
    "clay_pot": Recipe(
        name="clay_pot",
        inputs={"clay": 2},
        outputs={"clay_pot": 1},
        skill_requirement=15,
    ),
    "cooked_meat": Recipe(
        name="cooked_meat",
        inputs={"raw_meat": 1, "firewood": 0.5},
        outputs={"cooked_meat": 1},
        activity="cook_food",
        skill_requirement=0,
    ),
    "dried_meat": Recipe(
        name="dried_meat",
        inputs={"raw_meat": 2, "firewood": 1},
        outputs={"dried_meat": 1.5},
        activity="preserve_food",
        skill_requirement=5,
    ),
    "dried_fish": Recipe(
        name="dried_fish",
        inputs={"fish": 2, "firewood": 0.5},
        outputs={"dried_fish": 1.5},
        activity="preserve_food",
        skill_requirement=3,
    ),
    "bread": Recipe(
        name="bread",
        inputs={"grain": 2, "firewood": 0.5},
        outputs={"bread": 2},
        activity="cook_food",
        skill_requirement=10,
    ),
    "cloth": Recipe(
        name="cloth",
        inputs={"plant_fiber": 5},
        outputs={"cloth": 1},
        skill_requirement=15,
    ),
    "clothing": Recipe(
        name="clothing",
        inputs={"cloth": 2, "animal_hide": 1},
        outputs={"clothing": 1},
        tool_requirements=["knife"],
        skill_requirement=20,
    ),
    "tanned_leather": Recipe(
        name="tanned_leather",
        inputs={"animal_hide": 2},
        outputs={"tanned_leather": 1},
        skill_requirement=15,
    ),
}


def get_craftable_recipes(
    inventory: "Inventory",  # noqa: F821
    skill_level: float,
    activity_type: str = "craft_tools",
) -> list[Recipe]:
    """Return all recipes the villager can currently craft."""
    return [
        recipe for recipe in RECIPES.values()
        if recipe.activity == activity_type and recipe.can_craft(inventory, skill_level)
    ]


def execute_craft(
    recipe: Recipe,
    inventory: "Inventory",  # noqa: F821
    skill_level: float,
    quality_roll: float = 0.5,
) -> dict[str, float]:
    """
    Execute a crafting recipe: consume inputs, produce outputs.
    Returns dict of produced items and quantities.
    """
    # Consume inputs
    for item_type, qty in recipe.inputs.items():
        inventory.remove(item_type, qty)

    # Calculate output quality
    from village_sim.economy.inventory import create_item
    output_quality = 0.5
    if recipe.quality_from_skill:
        output_quality = min(1.0, 0.3 + 0.7 * (skill_level / 100.0) * quality_roll)

    # Produce outputs
    produced: dict[str, float] = {}
    for item_type, qty in recipe.outputs.items():
        item = create_item(item_type, qty, output_quality)
        inventory.add(item)
        produced[item_type] = qty

    return produced

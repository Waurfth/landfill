"""All productive activities with trait requirements, time costs, and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from numpy.random import Generator

from village_sim.world.resources import ResourceType


@dataclass
class Activity:
    """A productive activity a villager can perform."""

    name: str
    description: str = ""
    trait_weights: dict[str, float] = field(default_factory=dict)
    base_success_chance: float = 0.5
    base_hours: float = 4.0
    required_tools: list[str] = field(default_factory=list)
    required_season: Optional[list[str]] = None
    resource_type: Optional[ResourceType] = None
    outputs: dict[str, float] = field(default_factory=dict)
    danger_level: float = 0.0
    min_group_size: int = 1
    group_bonus: float = 0.0
    xp_category: str = ""
    fatigue_cost: float = 0.05

    def calculate_success(
        self,
        villager: "Villager",  # noqa: F821
        tool_quality: float = 1.0,
        group_size: int = 1,
        group_skill_avg: float = 0.0,
        weather_modifier: float = 1.0,
    ) -> float:
        """Calculate success probability for this villager."""
        # Trait-weighted score
        trait_score = weighted_trait_score(villager, self.trait_weights)

        # Skill modifier (0.5 to 2.0)
        category = self.xp_category or self.name
        skill = villager.memory.skill_level(category, villager.traits.intelligence)
        skill_mod = 0.5 + 1.5 * (skill / 100.0)

        # Tool modifier (0.5 to 1.5)
        tool_mod = 0.5 + tool_quality

        # Group modifier
        group_mod = 1.0
        if group_size > 1 and self.group_bonus > 0:
            # Diminishing returns per additional member
            for i in range(1, group_size):
                group_mod += self.group_bonus * (0.8 ** i)

        chance = (
            self.base_success_chance
            * trait_score
            * skill_mod
            * tool_mod
            * group_mod
            * weather_modifier
        )
        return max(0.05, min(0.95, chance))

    def calculate_yield(
        self,
        villager: "Villager",  # noqa: F821
        success_roll: float,
        tool_quality: float = 1.0,
    ) -> dict[str, float]:
        """Calculate output quantities on success."""
        if not self.outputs:
            return {}

        category = self.xp_category or self.name
        skill = villager.memory.skill_level(category, villager.traits.intelligence)
        skill_mult = 0.8 + 0.4 * (skill / 100.0)
        tool_mult = 0.9 + 0.2 * tool_quality

        result: dict[str, float] = {}
        for item_type, base_qty in self.outputs.items():
            qty = base_qty * skill_mult * tool_mult * (0.7 + success_roll * 0.3)
            result[item_type] = max(0.1, qty)
        return result


def weighted_trait_score(villager: "Villager", trait_weights: dict[str, float]) -> float:  # noqa: F821
    """
    Compute a score from villager's effective traits weighted by activity requirements.
    Returns a multiplier centered around 1.0 (0.5 = terrible, 1.5 = excellent).
    """
    if not trait_weights:
        return 1.0

    total_weight = sum(trait_weights.values())
    if total_weight == 0:
        return 1.0

    weighted_sum = 0.0
    for trait_name, weight in trait_weights.items():
        trait_val = villager.get_effective_trait(trait_name)
        weighted_sum += (trait_val / 100.0) * weight

    normalized = weighted_sum / total_weight  # 0 to 1, center ~0.5
    return 0.5 + normalized  # 0.5 to 1.5


# =============================================================================
# Activity definitions
# =============================================================================

ACTIVITIES: dict[str, Activity] = {
    "gather_berries": Activity(
        name="gather_berries",
        description="Gather wild berries and edible plants",
        trait_weights={"endurance": 0.4, "intelligence": 0.3, "dexterity": 0.3},
        base_success_chance=0.80,
        base_hours=3,
        resource_type=ResourceType.WILD_PLANTS,
        outputs={"berries": 6.0, "plant_fiber": 1.5},
        danger_level=0.02,
        fatigue_cost=0.05,
        xp_category="gathering",
    ),
    "hunt_small_game": Activity(
        name="hunt_small_game",
        description="Hunt rabbits, birds, and other small animals",
        trait_weights={"dexterity": 0.35, "patience": 0.25, "endurance": 0.2, "intelligence": 0.2},
        base_success_chance=0.65,
        base_hours=4,
        required_tools=["spear"],
        resource_type=ResourceType.GAME_SMALL,
        outputs={"raw_meat": 4.0, "animal_hide": 1.0},
        danger_level=0.05,
        fatigue_cost=0.08,
        xp_category="hunting",
    ),
    "hunt_large_game": Activity(
        name="hunt_large_game",
        description="Hunt deer, boar, and other large animals",
        trait_weights={"strength": 0.3, "endurance": 0.25, "risk_tolerance": 0.15, "dexterity": 0.2, "patience": 0.1},
        base_success_chance=0.45,
        base_hours=8,
        required_tools=["spear"],
        resource_type=ResourceType.GAME_LARGE,
        outputs={"raw_meat": 12.0, "animal_hide": 3.0},
        danger_level=0.15,
        min_group_size=1,
        group_bonus=0.15,
        fatigue_cost=0.12,
        xp_category="hunting",
    ),
    "fishing": Activity(
        name="fishing",
        description="Catch fish from rivers and streams",
        trait_weights={"patience": 0.4, "dexterity": 0.3, "intelligence": 0.3},
        base_success_chance=0.70,
        base_hours=4,
        required_tools=["fishing"],
        resource_type=ResourceType.FISH,
        outputs={"fish": 5.0},
        danger_level=0.02,
        fatigue_cost=0.04,
        xp_category="fishing",
    ),
    "chop_wood": Activity(
        name="chop_wood",
        description="Fell trees and chop wood",
        trait_weights={"strength": 0.5, "endurance": 0.35, "dexterity": 0.15},
        base_success_chance=0.9,
        base_hours=4,
        required_tools=["axe"],
        resource_type=ResourceType.TIMBER,
        outputs={"timber": 2.0, "firewood": 3.0},
        danger_level=0.05,
        fatigue_cost=0.10,
        xp_category="woodcutting",
    ),
    "mine_stone": Activity(
        name="mine_stone",
        description="Quarry stone from rocky outcrops",
        trait_weights={"strength": 0.45, "endurance": 0.4, "dexterity": 0.15},
        base_success_chance=0.8,
        base_hours=6,
        required_tools=["mining"],
        resource_type=ResourceType.STONE,
        outputs={"stone": 3.0},
        danger_level=0.08,
        fatigue_cost=0.12,
        xp_category="mining",
    ),
    "mine_ore": Activity(
        name="mine_ore",
        description="Mine iron ore from deep deposits",
        trait_weights={"strength": 0.4, "endurance": 0.35, "intelligence": 0.15, "dexterity": 0.1},
        base_success_chance=0.5,
        base_hours=7,
        required_tools=["mining"],
        resource_type=ResourceType.IRON_ORE,
        outputs={"iron_ore": 1.5},
        danger_level=0.12,
        fatigue_cost=0.14,
        xp_category="mining",
    ),
    "farm_plant": Activity(
        name="farm_plant",
        description="Prepare soil and plant seeds",
        trait_weights={"patience": 0.3, "endurance": 0.3, "strength": 0.2, "intelligence": 0.2},
        base_success_chance=0.8,
        base_hours=6,
        required_tools=["farming"],
        required_season=["spring", "summer"],
        resource_type=ResourceType.FARMLAND,
        outputs={},
        danger_level=0.01,
        fatigue_cost=0.08,
        xp_category="farming",
    ),
    "farm_tend": Activity(
        name="farm_tend",
        description="Weed, water, and care for crops",
        trait_weights={"patience": 0.3, "conscientiousness": 0.3, "endurance": 0.2, "intelligence": 0.2},
        base_success_chance=0.9,
        base_hours=4,
        required_tools=["farming"],
        required_season=["spring", "summer"],
        outputs={},
        danger_level=0.01,
        fatigue_cost=0.06,
        xp_category="farming",
    ),
    "farm_harvest": Activity(
        name="farm_harvest",
        description="Harvest mature crops",
        trait_weights={"endurance": 0.4, "strength": 0.3, "dexterity": 0.3},
        base_success_chance=0.95,
        base_hours=8,
        required_tools=["farming"],
        required_season=["summer", "autumn"],
        outputs={"grain": 10.0, "vegetables": 5.0},
        danger_level=0.01,
        group_bonus=0.1,
        fatigue_cost=0.10,
        xp_category="farming",
    ),
    "cook_food": Activity(
        name="cook_food",
        description="Cook raw food into meals",
        trait_weights={"intelligence": 0.3, "dexterity": 0.3, "patience": 0.2, "creativity": 0.2},
        base_success_chance=0.8,
        base_hours=2,
        required_tools=["knife"],
        outputs={"cooked_meat": 1.0},
        danger_level=0.02,
        fatigue_cost=0.03,
        xp_category="cooking",
    ),
    "preserve_food": Activity(
        name="preserve_food",
        description="Dry, smoke, or salt food for preservation",
        trait_weights={"intelligence": 0.3, "patience": 0.4, "conscientiousness": 0.3},
        base_success_chance=0.6,
        base_hours=4,
        required_tools=["knife"],
        outputs={"dried_meat": 1.0},
        danger_level=0.01,
        fatigue_cost=0.04,
        xp_category="cooking",
    ),
    "craft_tools": Activity(
        name="craft_tools",
        description="Create tools and useful items",
        trait_weights={"dexterity": 0.35, "intelligence": 0.35, "patience": 0.2, "creativity": 0.1},
        base_success_chance=0.65,
        base_hours=5,
        outputs={},
        danger_level=0.03,
        fatigue_cost=0.06,
        xp_category="crafting",
    ),
    "build_shelter": Activity(
        name="build_shelter",
        description="Construct or improve shelter structures",
        trait_weights={"strength": 0.35, "intelligence": 0.25, "dexterity": 0.2, "endurance": 0.2},
        base_success_chance=0.7,
        base_hours=8,
        required_tools=["axe", "construction"],
        outputs={},
        danger_level=0.06,
        group_bonus=0.2,
        fatigue_cost=0.12,
        xp_category="construction",
    ),
    "build_road": Activity(
        name="build_road",
        description="Clear and improve paths between locations",
        trait_weights={"strength": 0.4, "endurance": 0.4, "conscientiousness": 0.2},
        base_success_chance=0.9,
        base_hours=6,
        required_tools=["construction"],
        outputs={},
        danger_level=0.03,
        min_group_size=2,
        group_bonus=0.25,
        fatigue_cost=0.11,
        xp_category="construction",
    ),
    "gather_herbs": Activity(
        name="gather_herbs",
        description="Search for medicinal herbs and plants",
        trait_weights={"intelligence": 0.4, "patience": 0.3, "dexterity": 0.3},
        base_success_chance=0.4,
        base_hours=4,
        resource_type=ResourceType.MEDICINAL_HERBS,
        outputs={"medicine": 1.0},
        danger_level=0.02,
        fatigue_cost=0.04,
        xp_category="herbalism",
    ),
    "heal_villager": Activity(
        name="heal_villager",
        description="Treat an injured or sick villager",
        trait_weights={"intelligence": 0.4, "empathy": 0.3, "dexterity": 0.2, "patience": 0.1},
        base_success_chance=0.4,
        base_hours=3,
        outputs={},
        danger_level=0.01,
        fatigue_cost=0.04,
        xp_category="herbalism",
    ),
    "rest": Activity(
        name="rest",
        description="Take a deliberate rest day",
        base_success_chance=1.0,
        base_hours=0,
        outputs={},
        danger_level=0.0,
        fatigue_cost=-0.3,
        xp_category="",
    ),
    "socialize": Activity(
        name="socialize",
        description="Spend time socializing with other villagers",
        trait_weights={"sociability": 0.5, "empathy": 0.3, "intelligence": 0.2},
        base_success_chance=1.0,
        base_hours=4,
        outputs={},
        danger_level=0.0,
        min_group_size=2,
        fatigue_cost=0.02,
        xp_category="",
    ),
    "explore": Activity(
        name="explore",
        description="Explore the surrounding area for new resources",
        trait_weights={"risk_tolerance": 0.3, "endurance": 0.3, "intelligence": 0.2, "dexterity": 0.2},
        base_success_chance=0.3,
        base_hours=8,
        outputs={},
        danger_level=0.10,
        fatigue_cost=0.10,
        xp_category="exploration",
    ),
}

# =============================================================================
# Activity-to-need mapping: which needs does each activity help satisfy?
# =============================================================================

ACTIVITY_NEED_MAPPING: dict[str, list[str]] = {
    "gather_berries": ["hunger"],
    "hunt_small_game": ["hunger"],
    "hunt_large_game": ["hunger"],
    "fishing": ["hunger"],
    "chop_wood": ["warmth", "shelter"],
    "mine_stone": ["shelter"],
    "mine_ore": ["purpose"],
    "farm_plant": ["hunger"],
    "farm_tend": ["hunger"],
    "farm_harvest": ["hunger"],
    "cook_food": ["hunger", "comfort"],
    "preserve_food": ["hunger"],
    "craft_tools": ["purpose", "safety"],
    "build_shelter": ["shelter", "warmth", "safety"],
    "build_road": ["purpose"],
    "gather_herbs": ["health"],
    "heal_villager": ["health"],
    "rest": ["rest"],
    "socialize": ["social"],
    "explore": ["purpose", "safety"],
}

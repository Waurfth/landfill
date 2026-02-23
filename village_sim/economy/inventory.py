"""Item definitions, personal/family/community inventories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from village_sim.core.config import (
    CARRY_CAPACITY_BASE,
    COMMUNITY_INVENTORY_CAPACITY,
    FAMILY_INVENTORY_CAPACITY,
)


# =============================================================================
# Item catalog — all items and their properties
# =============================================================================

ITEM_CATALOG: dict[str, dict] = {
    # Food
    "berries": {"weight": 0.5, "perishable": True, "perish_days": 5, "food_value": 0.5},
    "raw_meat": {"weight": 2.0, "perishable": True, "perish_days": 3, "food_value": 1.0},
    "cooked_meat": {"weight": 1.5, "perishable": True, "perish_days": 7, "food_value": 1.5},
    "dried_meat": {"weight": 1.0, "perishable": True, "perish_days": 60, "food_value": 1.2},
    "grain": {"weight": 1.0, "perishable": True, "perish_days": 180, "food_value": 0.8},
    "bread": {"weight": 0.5, "perishable": True, "perish_days": 5, "food_value": 1.0},
    "fish": {"weight": 1.5, "perishable": True, "perish_days": 2, "food_value": 1.0},
    "dried_fish": {"weight": 0.8, "perishable": True, "perish_days": 90, "food_value": 1.0},
    "vegetables": {"weight": 1.0, "perishable": True, "perish_days": 10, "food_value": 0.5},
    # Raw materials
    "timber": {"weight": 10.0, "perishable": False},
    "stone": {"weight": 15.0, "perishable": False},
    "clay": {"weight": 5.0, "perishable": False},
    "iron_ore": {"weight": 12.0, "perishable": False},
    "plant_fiber": {"weight": 0.5, "perishable": False},
    "animal_hide": {"weight": 3.0, "perishable": True, "perish_days": 30},
    "tanned_leather": {"weight": 2.0, "perishable": False},
    "firewood": {"weight": 5.0, "perishable": False},
    # Tools
    "stone_axe": {"weight": 2.0, "perishable": False, "tool_type": "axe", "max_durability": 50},
    "stone_knife": {"weight": 0.5, "perishable": False, "tool_type": "knife", "max_durability": 40},
    "wooden_spear": {"weight": 2.0, "perishable": False, "tool_type": "spear", "max_durability": 30},
    "fishing_rod": {"weight": 1.0, "perishable": False, "tool_type": "fishing", "max_durability": 40},
    "bow": {"weight": 1.5, "perishable": False, "tool_type": "ranged", "max_durability": 60},
    "arrows": {"weight": 0.1, "perishable": False, "tool_type": "ammo"},
    "hoe": {"weight": 2.5, "perishable": False, "tool_type": "farming", "max_durability": 50},
    "hammer": {"weight": 3.0, "perishable": False, "tool_type": "construction", "max_durability": 70},
    "pickaxe": {"weight": 4.0, "perishable": False, "tool_type": "mining", "max_durability": 60},
    # Crafted goods
    "rope": {"weight": 1.0, "perishable": False},
    "basket": {"weight": 0.5, "perishable": False},
    "clay_pot": {"weight": 2.0, "perishable": False},
    "cloth": {"weight": 0.5, "perishable": False},
    "clothing": {"weight": 1.0, "perishable": False, "warmth_value": 0.3},
    "medicine": {"weight": 0.2, "perishable": True, "perish_days": 30, "heal_value": 0.3},
}


# Helper: get all food items
def food_items() -> list[str]:
    """Return all item types that have a food_value."""
    return [k for k, v in ITEM_CATALOG.items() if "food_value" in v]


# Helper: get tool type for an item
def get_tool_type(item_type: str) -> Optional[str]:
    """Return the tool_type for an item, or None."""
    return ITEM_CATALOG.get(item_type, {}).get("tool_type")


@dataclass
class Item:
    """A stack of items in an inventory."""

    item_type: str
    quantity: float = 1.0
    quality: float = 0.5          # 0-1, affects effectiveness
    days_since_created: int = 0
    current_durability: float = 0.0   # for tools
    max_durability: float = 0.0       # for tools

    @property
    def is_perishable(self) -> bool:
        return ITEM_CATALOG.get(self.item_type, {}).get("perishable", False)

    @property
    def perish_days(self) -> int:
        return ITEM_CATALOG.get(self.item_type, {}).get("perish_days", 9999)

    @property
    def weight_per_unit(self) -> float:
        return ITEM_CATALOG.get(self.item_type, {}).get("weight", 1.0)

    @property
    def total_weight(self) -> float:
        return self.quantity * self.weight_per_unit

    @property
    def food_value(self) -> float:
        return ITEM_CATALOG.get(self.item_type, {}).get("food_value", 0.0)

    @property
    def is_spoiled(self) -> bool:
        return self.is_perishable and self.days_since_created >= self.perish_days

    @property
    def is_tool(self) -> bool:
        return "tool_type" in ITEM_CATALOG.get(self.item_type, {})

    @property
    def tool_type(self) -> Optional[str]:
        return get_tool_type(self.item_type)

    @property
    def tool_quality(self) -> float:
        """Effective quality of a tool, degraded by durability."""
        if not self.is_tool or self.max_durability == 0:
            return self.quality
        return self.quality * (self.current_durability / self.max_durability)

    def age_one_day(self) -> None:
        self.days_since_created += 1


def create_item(item_type: str, quantity: float = 1.0, quality: float = 0.5) -> Item:
    """Factory to create an Item with correct durability from catalog."""
    cat = ITEM_CATALOG.get(item_type, {})
    max_dur = cat.get("max_durability", 0.0)
    return Item(
        item_type=item_type,
        quantity=quantity,
        quality=quality,
        current_durability=max_dur,
        max_durability=max_dur,
    )


class Inventory:
    """Container for items with weight capacity."""

    def __init__(self, capacity: float = CARRY_CAPACITY_BASE, owner_type: str = "personal") -> None:
        self.items: dict[str, list[Item]] = {}
        self.capacity = capacity
        self.owner_type = owner_type

    def add(self, item: Item) -> bool:
        """Add an item. Returns False if over capacity."""
        if self.total_weight() + item.total_weight > self.capacity:
            # Try to add partial
            available = self.capacity - self.total_weight()
            if available <= 0:
                return False
            addable = available / item.weight_per_unit
            if addable < 0.01:
                return False
            item.quantity = addable

        if item.item_type not in self.items:
            self.items[item.item_type] = []

        # Try to merge with existing stack of similar quality
        for existing in self.items[item.item_type]:
            if abs(existing.quality - item.quality) < 0.05 and not item.is_tool:
                existing.quantity += item.quantity
                return True

        self.items[item.item_type].append(item)
        return True

    def remove(self, item_type: str, quantity: float) -> Optional[Item]:
        """Remove quantity of item_type. Returns the removed Item or None."""
        if item_type not in self.items:
            return None

        stacks = self.items[item_type]
        removed_qty = 0.0
        avg_quality = 0.0

        # Remove from oldest stacks first
        while removed_qty < quantity and stacks:
            stack = stacks[0]
            take = min(quantity - removed_qty, stack.quantity)
            avg_quality = (avg_quality * removed_qty + stack.quality * take) / (removed_qty + take) if (removed_qty + take) > 0 else 0
            removed_qty += take
            stack.quantity -= take
            if stack.quantity <= 0.001:
                stacks.pop(0)

        if removed_qty <= 0:
            return None

        # Clean up empty lists
        if not stacks:
            del self.items[item_type]

        return Item(item_type=item_type, quantity=removed_qty, quality=avg_quality)

    def has(self, item_type: str, min_quantity: float = 1.0) -> bool:
        return self.total_of(item_type) >= min_quantity

    def total_of(self, item_type: str) -> float:
        if item_type not in self.items:
            return 0.0
        return sum(s.quantity for s in self.items[item_type])

    def total_weight(self) -> float:
        return sum(
            s.total_weight for stacks in self.items.values() for s in stacks
        )

    def remaining_capacity(self) -> float:
        return max(0.0, self.capacity - self.total_weight())

    def get_best_tool(self, tool_type: str) -> Optional[Item]:
        """Find the best tool of a given type."""
        best: Optional[Item] = None
        for stacks in self.items.values():
            for item in stacks:
                if item.tool_type == tool_type and item.current_durability > 0:
                    if best is None or item.tool_quality > best.tool_quality:
                        best = item
        return best

    def has_tool_type(self, tool_type: str) -> bool:
        """Check if inventory contains a functional tool of this type."""
        return self.get_best_tool(tool_type) is not None

    def daily_perish(self) -> list[Item]:
        """Age items and remove spoiled ones. Returns spoiled items."""
        spoiled: list[Item] = []
        for item_type in list(self.items.keys()):
            surviving: list[Item] = []
            for item in self.items[item_type]:
                item.age_one_day()
                if item.is_spoiled:
                    spoiled.append(item)
                else:
                    surviving.append(item)
            if surviving:
                self.items[item_type] = surviving
            else:
                del self.items[item_type]
        return spoiled

    def get_all_food(self) -> list[Item]:
        """Get all food items sorted by perish urgency."""
        result: list[Item] = []
        for item_type in food_items():
            if item_type in self.items:
                result.extend(self.items[item_type])
        result.sort(key=lambda i: i.perish_days - i.days_since_created)
        return result

    def total_food_value(self) -> float:
        """Total food value across all food items."""
        total = 0.0
        for item_type in food_items():
            for item in self.items.get(item_type, []):
                total += item.quantity * item.food_value
        return total


class FamilyInventory(Inventory):
    """Shared inventory for a family unit."""

    def __init__(self, family_id: int) -> None:
        super().__init__(capacity=FAMILY_INVENTORY_CAPACITY, owner_type="family")
        self.family_id = family_id


class CommunityInventory(Inventory):
    """Shared inventory for the entire village."""

    def __init__(self) -> None:
        super().__init__(capacity=COMMUNITY_INVENTORY_CAPACITY, owner_type="community")

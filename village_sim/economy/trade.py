"""Bilateral barter system with subjective valuation.

No money exists — value is subjective and depends on each villager's current
needs, inventory, personality, and experience.  Trade emerges from surplus/deficit
mismatches between villagers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from numpy.random import Generator

from village_sim.core.config import (
    BASE_DAILY_FOOD_NEED,
    TRADE_DEFICIT_DAYS_THRESHOLD,
    TRADE_DIMINISHING_SURPLUS_FACTOR,
    TRADE_MAX_ROUNDS_PER_DAY,
    TRADE_PERSONALITY_MARGIN,
    TRADE_SURPLUS_DAYS_THRESHOLD,
    TRADE_TRUST_WEIGHT,
    TRADE_VALUE_FOOD_HUNGRY_MULTIPLIER,
    TRADE_WILLINGNESS_BASE,
)
from village_sim.economy.inventory import ITEM_CATALOG, Inventory, Item, create_item, food_items


# =============================================================================
# Trade offer
# =============================================================================

@dataclass
class TradeOffer:
    """A proposed bilateral trade."""

    offering: dict[str, float]      # item_type -> quantity offered
    requesting: dict[str, float]    # item_type -> quantity requested
    offerer_id: int = -1
    target_id: int = -1


@dataclass
class TradeRecord:
    """Record of a completed trade for metrics."""

    day: int
    offerer_id: int
    target_id: int
    items_offered: dict[str, float]
    items_received: dict[str, float]


# =============================================================================
# Subjective value calculation
# =============================================================================

def subjective_value(
    villager: "Villager",  # noqa: F821
    item_type: str,
    quantity: float,
    family_inv: Optional[Inventory] = None,
) -> float:
    """How much does this villager value this item RIGHT NOW?

    Based on:
    - Current need satisfaction (food worth more when hungry)
    - Current inventory (diminishing marginal value)
    - Personality (ambitious villagers value tools more, etc.)
    - Experience (skilled hunter values raw meat less)

    Returns abstract utility units — only meaningful for comparison within
    this villager's perspective.
    """
    cat = ITEM_CATALOG.get(item_type, {})
    base_weight = cat.get("weight", 1.0)

    # Start with a base value proportional to weight (proxy for effort to obtain)
    base_value = base_weight * 0.5

    # -- Food value adjustment --
    food_val = cat.get("food_value", 0.0)
    if food_val > 0:
        hunger_sat = villager.needs.needs["hunger"].satisfaction
        # Food is worth more when hungry (exponential)
        if hunger_sat < 0.5:
            hunger_mult = 1.0 + (0.5 - hunger_sat) * (TRADE_VALUE_FOOD_HUNGRY_MULTIPLIER - 1.0) * 2
        else:
            hunger_mult = 1.0
        base_value = food_val * hunger_mult * 2.0

        # Shelf life bonus: longer-lasting food is worth more
        perish_days = cat.get("perish_days", 999)
        if perish_days < 999:
            shelf_bonus = min(1.0, perish_days / 60.0)
            base_value *= (0.7 + 0.3 * shelf_bonus)

    # -- Tool value adjustment --
    tool_type = cat.get("tool_type")
    if tool_type:
        # Tools are very valuable if you don't have one
        has_tool = False
        if villager.personal_inventory:
            has_tool = villager.personal_inventory.has_tool_type(tool_type)
        if not has_tool and family_inv:
            has_tool = family_inv.has_tool_type(tool_type)

        if not has_tool:
            base_value *= 3.0  # much more valuable when you lack it
        else:
            base_value *= 0.8  # still useful as backup

        # Ambitious villagers value tools more
        base_value *= (0.8 + 0.4 * villager.traits.ambition / 100.0)

    # -- Warmth value (clothing) --
    warmth_val = cat.get("warmth_value", 0.0)
    if warmth_val > 0:
        warmth_sat = villager.needs.needs["warmth"].satisfaction
        if warmth_sat < 0.5:
            base_value *= 2.0

    # -- Medicine value --
    heal_val = cat.get("heal_value", 0.0)
    if heal_val > 0:
        health_sat = villager.needs.needs["health"].satisfaction
        if health_sat < 0.7:
            base_value *= 2.5

    # -- Diminishing marginal value for items already owned --
    owned = 0.0
    if family_inv:
        owned += family_inv.total_of(item_type)
    if villager.personal_inventory:
        owned += villager.personal_inventory.total_of(item_type)

    if owned > 0:
        # More of the same item is worth less
        diminish = 1.0 / (1.0 + owned * TRADE_DIMINISHING_SURPLUS_FACTOR)
        base_value *= diminish

    # -- Skill-based adjustment --
    # If the villager can easily obtain this item, it's worth less to them
    if item_type in ("raw_meat", "fish", "berries", "vegetables"):
        skill_map = {
            "raw_meat": "hunting",
            "fish": "fishing",
            "berries": "gathering",
            "vegetables": "farming",
        }
        skill_name = skill_map.get(item_type, "")
        if skill_name:
            skill = villager.memory.skill_level(skill_name, villager.traits.intelligence)
            if skill > 30:
                # High skill = can get it easily = less valuable
                base_value *= max(0.5, 1.0 - skill / 200.0)

    return max(0.01, base_value * quantity)


# =============================================================================
# Surplus / deficit calculation
# =============================================================================

def _get_surplus(
    villager: "Villager",  # noqa: F821
    inv: Inventory,
) -> dict[str, float]:
    """Items beyond what this villager needs for the next N days."""
    surplus: dict[str, float] = {}
    food_need = BASE_DAILY_FOOD_NEED * TRADE_SURPLUS_DAYS_THRESHOLD

    # Track remaining food need
    remaining_food_need = food_need
    for item_type, stacks in inv.items.items():
        total_qty = sum(s.quantity for s in stacks)
        if total_qty <= 0.01:
            continue

        cat = ITEM_CATALOG.get(item_type, {})
        food_val = cat.get("food_value", 0.0)

        if food_val > 0:
            # Keep enough food for N days
            needed_qty = remaining_food_need / max(0.01, food_val)
            excess = total_qty - needed_qty
            if excess > 0.5:
                surplus[item_type] = excess
            remaining_food_need -= min(total_qty, needed_qty) * food_val
        elif cat.get("tool_type"):
            # Keep one of each tool type, surplus the rest
            if total_qty > 1:
                surplus[item_type] = total_qty - 1
        else:
            # Raw materials: surplus anything above 5 units
            if total_qty > 5:
                surplus[item_type] = total_qty - 5

    return surplus


def _get_deficits(
    villager: "Villager",  # noqa: F821
    inv: Inventory,
) -> dict[str, float]:
    """Items below what this villager needs for the next N days."""
    deficits: dict[str, float] = {}
    food_need = BASE_DAILY_FOOD_NEED * TRADE_DEFICIT_DAYS_THRESHOLD

    # Check total food
    total_food_value = inv.total_food_value()
    if total_food_value < food_need:
        # Need more food — prefer long-lasting items
        deficit_food = food_need - total_food_value
        for ft in ("grain", "dried_meat", "dried_fish", "cooked_meat", "bread", "berries"):
            cat = ITEM_CATALOG.get(ft, {})
            fv = cat.get("food_value", 0.5)
            deficits[ft] = deficit_food / fv
            break  # just the top preference

    # Check tools
    needed_tools = {"axe", "knife", "spear", "farming", "fishing"}
    for tool_type in needed_tools:
        has = False
        if inv:
            has = inv.has_tool_type(tool_type)
        if villager.personal_inventory and not has:
            has = villager.personal_inventory.has_tool_type(tool_type)
        if not has:
            # Find an item with this tool_type
            for itype, cat in ITEM_CATALOG.items():
                if cat.get("tool_type") == tool_type:
                    deficits[itype] = 1.0
                    break

    # Check warmth/clothing
    warmth_sat = villager.needs.needs["warmth"].satisfaction
    if warmth_sat < 0.4:
        if not inv.has("clothing"):
            deficits["clothing"] = 1.0

    # Check medicine
    if villager.health < 70:
        if not inv.has("medicine"):
            deficits["medicine"] = 1.0

    return deficits


# =============================================================================
# Trade system
# =============================================================================

class TradeSystem:
    """Bilateral barter with subjective valuation."""

    def __init__(self, rng: Generator) -> None:
        self._rng = rng
        self.daily_trades: list[TradeRecord] = []
        self.total_trades: int = 0
        self.total_items_exchanged: float = 0.0

    def reset_daily(self) -> None:
        """Reset daily trade tracking."""
        self.daily_trades.clear()

    def generate_offer(
        self,
        villager: "Villager",  # noqa: F821
        partner: "Villager",  # noqa: F821
        villager_inv: Inventory,
        partner_inv_estimate: Inventory,
        relationship_trust: float,
    ) -> Optional[TradeOffer]:
        """Generate a trade offer from villager to partner.

        Villager doesn't know partner's exact inventory but estimates based
        on relationship closeness and observation.
        """
        my_surplus = _get_surplus(villager, villager_inv)
        partner_deficit_estimate = _get_deficits(partner, partner_inv_estimate)

        if not my_surplus:
            return None

        # Find items I have in surplus that the partner might want
        offering: dict[str, float] = {}
        requesting: dict[str, float] = {}

        # Offer surplus items that match partner's estimated needs
        offer_value = 0.0
        for item_type, qty in my_surplus.items():
            if item_type in partner_deficit_estimate:
                offer_qty = min(qty, partner_deficit_estimate[item_type])
                if offer_qty > 0.01:
                    offering[item_type] = offer_qty
                    offer_value += subjective_value(partner, item_type, offer_qty, partner_inv_estimate)

        # If no match found, offer most surplus item
        if not offering and my_surplus:
            best_item = max(my_surplus.keys(), key=lambda k: my_surplus[k])
            offer_qty = min(my_surplus[best_item], my_surplus[best_item] * 0.5)
            if offer_qty > 0.01:
                offering[best_item] = offer_qty
                offer_value += subjective_value(partner, best_item, offer_qty, partner_inv_estimate)

        if not offering or offer_value < 0.1:
            return None

        # Request items I need from what partner might have
        my_deficits = _get_deficits(villager, villager_inv)
        request_value = 0.0

        for item_type, qty in my_deficits.items():
            est_partner_has = partner_inv_estimate.total_of(item_type)
            if est_partner_has > 0:
                req_qty = min(qty, est_partner_has * 0.5)  # don't ask for everything
                if req_qty > 0.01:
                    requesting[item_type] = req_qty
                    request_value += subjective_value(villager, item_type, req_qty, villager_inv)

        # If no specific deficit, request what the partner has in surplus
        if not requesting:
            partner_surplus = _get_surplus(partner, partner_inv_estimate)
            for item_type, qty in partner_surplus.items():
                if item_type not in offering:
                    req_qty = min(qty * 0.3, qty)
                    if req_qty > 0.01:
                        requesting[item_type] = req_qty
                        request_value += subjective_value(villager, item_type, req_qty, villager_inv)
                        break

        if not requesting:
            return None

        # Personality adjusts the deal — ambitious villagers ask for more
        ambition_factor = 1.0 + (villager.traits.ambition - 50) / 100.0 * TRADE_PERSONALITY_MARGIN
        # Scale request to roughly match offer value, adjusted by personality
        if request_value > 0 and offer_value > 0:
            value_ratio = (offer_value * ambition_factor) / request_value
            for item_type in requesting:
                requesting[item_type] = min(
                    requesting[item_type] * value_ratio,
                    requesting[item_type],
                )

        return TradeOffer(
            offering=offering,
            requesting=requesting,
            offerer_id=villager.id,
            target_id=partner.id,
        )

    def evaluate_offer(
        self,
        villager: "Villager",  # noqa: F821
        offer: TradeOffer,
        villager_inv: Inventory,
        trust: float,
    ) -> bool:
        """Would accepting this trade improve the villager's situation?

        Compares subjective value of what they'd get vs what they'd give.
        Trust and personality affect the threshold.
        """
        # Value of what I'd receive
        receive_value = sum(
            subjective_value(villager, item_type, qty, villager_inv)
            for item_type, qty in offer.offering.items()
        )

        # Value of what I'd give up
        give_value = sum(
            subjective_value(villager, item_type, qty, villager_inv)
            for item_type, qty in offer.requesting.items()
        )

        if give_value <= 0:
            return receive_value > 0

        # Trust factor: more trust = more willing to accept marginal deals
        trust_bonus = trust * TRADE_TRUST_WEIGHT

        # Personality: agreeable/empathetic villagers accept worse deals
        agreeableness = (villager.traits.empathy + villager.traits.sociability) / 200.0
        personality_bonus = agreeableness * TRADE_PERSONALITY_MARGIN

        # Require at least this ratio of receive/give to accept
        threshold = max(0.5, 1.0 - trust_bonus - personality_bonus)

        ratio = receive_value / max(0.01, give_value)
        return ratio >= threshold

    def execute_trade(
        self,
        offer: TradeOffer,
        offerer: "Villager",  # noqa: F821
        target: "Villager",  # noqa: F821
        offerer_inv: Inventory,
        target_inv: Inventory,
        day: int,
    ) -> bool:
        """Transfer items between inventories.

        Returns True if the trade was successfully executed.
        """
        # Verify both parties have the items
        for item_type, qty in offer.offering.items():
            if offerer_inv.total_of(item_type) < qty * 0.99:
                return False
        for item_type, qty in offer.requesting.items():
            if target_inv.total_of(item_type) < qty * 0.99:
                return False

        # Execute transfers
        # offerer gives -> target receives
        for item_type, qty in offer.offering.items():
            removed = offerer_inv.remove(item_type, qty)
            if removed:
                target_inv.add(removed)

        # target gives -> offerer receives
        for item_type, qty in offer.requesting.items():
            removed = target_inv.remove(item_type, qty)
            if removed:
                offerer_inv.add(removed)

        # Record the trade
        items_exchanged = sum(offer.offering.values()) + sum(offer.requesting.values())
        self.total_trades += 1
        self.total_items_exchanged += items_exchanged

        record = TradeRecord(
            day=day,
            offerer_id=offer.offerer_id,
            target_id=offer.target_id,
            items_offered=dict(offer.offering),
            items_received=dict(offer.requesting),
        )
        self.daily_trades.append(record)

        return True

    def estimate_partner_inventory(
        self,
        villager: "Villager",  # noqa: F821
        partner: "Villager",  # noqa: F821
        partner_actual_inv: Inventory,
        trust: float,
        familiarity: float,
    ) -> Inventory:
        """Estimate what a trading partner has in inventory.

        Better estimates come from closer relationships.
        The actual_inv is the partner's family inventory (ground truth).
        """
        estimate = Inventory(capacity=999.0)

        # Accuracy scales with familiarity and trust
        accuracy = max(0.2, min(0.9, (trust + familiarity) / 2.0))
        noise_factor = 1.0 - accuracy

        # Copy items with noise
        for item_type, stacks in partner_actual_inv.items.items():
            total = sum(s.quantity for s in stacks)
            if total > 0:
                # Add noise based on how well they know each other
                noisy_qty = total * (1.0 + float(self._rng.uniform(-noise_factor, noise_factor)))
                noisy_qty = max(0, noisy_qty)
                if noisy_qty > 0.1:
                    estimate.add(create_item(item_type, noisy_qty))

        return estimate

"""Main simulation loop: 14-step daily tick cycle."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from village_sim.agents.decision import DecisionEngine, WorldState
from village_sim.agents.villager import Villager, generate_initial_population
from village_sim.core.clock import SimClock
from village_sim.core.config import (
    INITIAL_POPULATION,
    PREGNANCY_DURATION_DAYS,
    STARTING_FOOD_PER_PERSON,
    STARTING_SHELTERS,
    STARTING_TOOLS,
    TOOL_DURABILITY_LOSS_PER_USE,
    VILLAGE_CENTER,
    WATER_AUTO_SATISFY_AMOUNT,
    WATER_PROXIMITY_RADIUS,
)
from village_sim.economy.activities import ACTIVITIES
from village_sim.economy.crafting import RECIPES, execute_craft, get_craftable_recipes
from village_sim.economy.inventory import (
    CommunityInventory,
    Inventory,
    create_item,
)
from village_sim.economy.trade import TradeSystem
from village_sim.simulation.events import EventSystem
from village_sim.simulation.metrics import MetricsCollector
from village_sim.social.family import FamilyManager
from village_sim.social.groups import GroupManager
from village_sim.social.influence import InfluenceSystem
from village_sim.social.relationships import RelationshipManager
from village_sim.viz.logger import SimLogger
from village_sim.world.climate import Climate
from village_sim.world.crops import CropManager
from village_sim.world.infrastructure import InfrastructureManager
from village_sim.world.map import WorldMap
from village_sim.world.pathfinding import estimate_travel_time
from village_sim.world.resources import ResourceManager, ResourceType


class SimulationEngine:
    """Orchestrates the entire village simulation."""

    def __init__(self, seed: int = 42, population: int = INITIAL_POPULATION) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self._population_size = population

        # Core systems
        self.clock = SimClock()
        self.world_map = WorldMap()
        self.resource_manager = ResourceManager()
        self.climate = Climate(self.rng)
        self.crop_manager = CropManager()
        self.infrastructure = InfrastructureManager()

        # Agents
        self.villagers: list[Villager] = []
        self.dead_villagers: list[Villager] = []
        self._villager_map: dict[int, Villager] = {}
        self._next_villager_id: int = 0

        # Social
        self.family_manager = FamilyManager()
        self.relationship_manager = RelationshipManager()
        self.group_manager = GroupManager()
        self.influence_system = InfluenceSystem()

        # Economy
        self.community_inventory = CommunityInventory()
        self.trade_system = TradeSystem(self.rng)

        # Decision engine
        self.decision_engine = DecisionEngine(self.rng)

        # Simulation systems
        self.event_system = EventSystem(self.rng)
        self.metrics = MetricsCollector()
        self.logger = SimLogger()

        # Dashboard callback (set externally)
        self._dashboard_callback = None

    def initialize(self) -> None:
        """Generate the world and populate the village."""
        # Generate world
        self.world_map.generate(self.rng)
        self.resource_manager.generate_resources(self.world_map, self.rng)

        # Generate population
        self.villagers = generate_initial_population(self._population_size, self.rng)
        self._next_villager_id = max(v.id for v in self.villagers) + 1
        self._villager_map = {v.id: v for v in self.villagers}

        # Build families
        self.family_manager.build_from_villagers(self.villagers)

        # Give each villager a personal inventory
        for v in self.villagers:
            v.personal_inventory = Inventory(capacity=30.0)

        # Starting conditions
        if STARTING_TOOLS:
            self._distribute_starting_tools()
        if STARTING_SHELTERS:
            self._create_starting_shelters()
        self._distribute_starting_food()

        # Initial climate
        self.climate.advance_day(self.clock.season, self.clock.day_of_season)

        self.logger.log("LIFECYCLE", f"Village founded with {len(self.villagers)} villagers")

    def set_dashboard_callback(self, callback) -> None:
        """Set a callback function for real-time dashboard updates."""
        self._dashboard_callback = callback

    def run(self, days: int) -> None:
        """Run the simulation for a number of days."""
        for _ in range(days):
            self.tick()
            if self._dashboard_callback:
                self._dashboard_callback(self.clock.day, self.metrics)

    def tick(self) -> None:
        """One day of simulation — the 14-step cycle."""
        alive = [v for v in self.villagers if v.is_alive]
        if not alive:
            return

        # 1. DAWN — World updates
        self.clock.advance()
        day = self.clock.day
        self.climate.advance_day(self.clock.season, self.clock.day_of_season)
        self.resource_manager.daily_regeneration(self.clock.season)
        self.crop_manager.daily_update(self.climate, set())

        # 2. DAWN — Random events
        events = self.event_system.check_random_events(
            day, self.clock.season, self.climate.current_weather, alive,
        )
        if events:
            self.event_system.apply_events(
                events, self._villager_map, self.family_manager, self.infrastructure,
            )
            for event in events:
                self.logger.log("EVENT", event.description, day=day)

        # 3. MORNING — Decisions
        world_state = self._build_world_state()
        daylight = self.clock.daylight_hours()

        for v in alive:
            if v.is_child and v.age_years < 6:
                v.current_activity = "rest"
                continue
            if v.recovery_days > 0:
                v.current_activity = "rest"
                continue

            schedule = self.decision_engine.plan_day(v, world_state, daylight)
            v.current_activity = schedule[0].activity_name if schedule else "rest"
            v._day_schedule = schedule  # stash for execution

        # 4. FORM WORK PARTIES
        self.group_manager.resolve_work_parties(
            alive, self.decision_engine, self.relationship_manager, self.rng,
        )
        for _ in self.group_manager.work_parties:
            self.metrics.record_work_party()

        # 5. DAYTIME — Execute activities
        tended_positions: set[tuple[int, int]] = set()
        for v in alive:
            schedule = getattr(v, "_day_schedule", None)
            if not schedule:
                continue
            self._execute_schedule(v, schedule, tended_positions, day)

        # 6. AFTERNOON — Auto thirst satisfaction
        self._auto_satisfy_thirst(alive)

        # 7. EVENING — Social phase
        self._resolve_social_phase(alive, day)

        # 7b. EVENING — Trade phase
        self._resolve_trade_phase(alive, day)

        # 8. NIGHT — Family phase
        for fam in self.family_manager.families.values():
            fam.distribute_food(self._villager_map)

        # 9. NIGHT — Need updates
        for v in alive:
            shelter_quality = self.infrastructure.shelter_quality_for(v.family_id)
            warmth_mod = self.climate.warmth_need_modifier()
            had_social = v.needs.needs["social"].satisfaction > 0.7
            was_productive = v.current_activity not in ("rest", "socialize", "")

            v.needs.daily_decay(
                warmth_modifier=warmth_mod,
                shelter_quality=shelter_quality,
                had_social_interaction=had_social,
                was_productive=was_productive,
            )
            v.daily_update(day, self.climate, self.rng)

        # 10. NIGHT — Sentiment contagion
        self.influence_system.spread_sentiment(self.villagers, self.relationship_manager)

        # 11. NIGHT — Lifecycle events (births, deaths, marriages)
        self._process_lifecycle(day)

        # 12. OVERNIGHT — Inventory maintenance (perishables)
        for v in alive:
            if v.personal_inventory:
                v.personal_inventory.daily_perish()
        for fam in self.family_manager.families.values():
            fam.inventory.daily_perish()

        # 13. OVERNIGHT — Infrastructure degradation
        self.infrastructure.daily_degradation(self.climate.shelter_damage_modifier())
        self.relationship_manager.daily_decay_all(day)

        # 14. METRICS & LOG
        self.metrics.collect_daily(
            day, self.villagers, self.family_manager, self.resource_manager,
        )
        self.logger.flush_day(day)

        # Cleanup
        for v in alive:
            if hasattr(v, "_day_schedule"):
                del v._day_schedule

    # ------------------------------------------------------------------
    # Activity execution
    # ------------------------------------------------------------------

    def _execute_schedule(
        self,
        villager: Villager,
        schedule: list,
        tended_positions: set[tuple[int, int]],
        day: int,
    ) -> None:
        """Execute a villager's daily activity schedule."""
        for plan in schedule:
            act = ACTIVITIES.get(plan.activity_name)
            if act is None:
                continue

            # Handle special activities
            if plan.activity_name == "rest":
                villager.fatigue = max(0, villager.fatigue + act.fatigue_cost)
                villager.needs.satisfy("rest", 0.4)
                continue

            if plan.activity_name == "socialize":
                villager.needs.satisfy("social", 0.3)
                continue

            # Travel to resource node
            target_pos = plan.target_position or villager.current_position
            if target_pos != villager.current_position:
                travel_time = self._quick_travel_estimate(villager.current_position, target_pos)
                villager.fatigue += travel_time * 0.05
                villager.memory.add_route_trip(villager.current_position, target_pos)
                villager.current_position = target_pos

            # Check for group work
            party = self.group_manager.get_party_for(villager.id)
            group_size = party.size if party else 1

            # Get best tool
            tool_quality = 1.0
            tool_item = None
            if act.required_tools:
                inv = villager.personal_inventory
                fam = self.family_manager.get_family(villager.family_id)
                for tool_type in act.required_tools:
                    tool_item = inv.get_best_tool(tool_type) if inv else None
                    if tool_item is None and fam:
                        tool_item = fam.inventory.get_best_tool(tool_type)
                    if tool_item:
                        tool_quality = tool_item.tool_quality
                        break
                if tool_item is None:
                    # No tool available — skip this activity
                    continue

            # Roll for success
            weather_mod = self.climate.outdoor_work_modifier()
            success_chance = act.calculate_success(
                villager, tool_quality, group_size,
                weather_modifier=weather_mod,
            )
            roll = float(self.rng.random())
            success = roll < success_chance

            # Handle crafting specially
            if plan.activity_name == "craft_tools":
                self._handle_crafting(villager, success, day)
            elif plan.activity_name == "cook_food":
                self._handle_cooking(villager, success, "cooked_meat", day)
            elif plan.activity_name == "preserve_food":
                self._handle_cooking(villager, success, "dried_meat", day)
            elif plan.activity_name in ("farm_plant", "farm_tend", "farm_harvest"):
                self._handle_farming(villager, plan.activity_name, success, tended_positions, day)
            elif plan.activity_name == "build_shelter":
                self._handle_building(villager, success, day)
            elif plan.activity_name == "explore":
                self._handle_explore(villager, success, day)
            elif plan.activity_name == "heal_villager":
                self._handle_healing(villager, success, day)
            elif success and act.outputs:
                # Standard resource-producing activity
                yields = act.calculate_yield(villager, roll, tool_quality)
                resource_node = None
                if plan.target_resource_id is not None:
                    resource_node = self.resource_manager.get_node(plan.target_resource_id)
                    # Fallback: if planned node is depleted, find a nearby alternative
                    if resource_node and resource_node.current_abundance <= 0 and act.resource_type:
                        alt = self.resource_manager.get_nearest_of_type(
                            villager.current_position, act.resource_type, rng=self.rng,
                        )
                        if alt and alt.current_abundance > 0:
                            resource_node = alt

                for item_type, qty in yields.items():
                    if resource_node:
                        actual = resource_node.harvest(qty, tool_quality)
                    else:
                        actual = qty
                    if actual > 0:
                        item = create_item(item_type, actual)
                        # Add to family inventory (larger capacity)
                        fam = self.family_manager.get_family(villager.family_id)
                        if fam:
                            fam.inventory.add(item)
                        elif villager.personal_inventory:
                            villager.personal_inventory.add(item)

                self.logger.log(
                    "ACTIVITY",
                    f"{villager.name} {act.description}, yielded {yields}",
                    villager_ids=[villager.id],
                    day=day,
                )

            # XP gain
            xp_cat = act.xp_category or act.name
            if xp_cat:
                villager.memory.add_experience(xp_cat, success, villager.traits.intelligence)

            # Fatigue
            villager.fatigue += act.fatigue_cost * plan.planned_hours
            villager.fatigue = min(1.0, villager.fatigue)

            # Tool durability
            if tool_item and tool_item.max_durability > 0:
                tool_item.current_durability -= TOOL_DURABILITY_LOSS_PER_USE
                if tool_item.current_durability <= 0:
                    self.logger.log(
                        "ACTIVITY", f"{villager.name}'s {tool_item.item_type} broke",
                        villager_ids=[villager.id], day=day,
                    )

            # Danger check
            if act.danger_level > 0 and self.rng.random() < act.danger_level:
                damage = float(self.rng.uniform(5, 20))
                villager.health = max(0, villager.health - damage)
                villager.memory.add_event(day, f"injured during {act.name}", -0.3)
                self.logger.log(
                    "ACTIVITY",
                    f"{villager.name} was injured during {act.name} (-{damage:.0f} health)",
                    villager_ids=[villager.id], day=day,
                )

            # Purpose satisfaction from productive work
            villager.needs.satisfy("purpose", 0.1)

    def _handle_crafting(self, villager: Villager, success: bool, day: int) -> None:
        """Handle crafting activity — pick best recipe and craft."""
        inv = villager.personal_inventory
        fam = self.family_manager.get_family(villager.family_id)
        source_inv = fam.inventory if fam else inv
        if source_inv is None:
            return

        skill = villager.memory.skill_level("crafting", villager.traits.intelligence)
        recipes = get_craftable_recipes(source_inv, skill, "craft_tools")
        if not recipes:
            return

        # Pick most useful recipe (prefer tools they lack)
        recipe = recipes[0]
        for r in recipes:
            for out_type in r.outputs:
                if not source_inv.has(out_type):
                    recipe = r
                    break

        if success:
            produced = execute_craft(recipe, source_inv, skill, float(self.rng.random()))
            self.logger.log(
                "ACTIVITY", f"{villager.name} crafted {produced}",
                villager_ids=[villager.id], day=day,
            )

    def _handle_cooking(self, villager: Villager, success: bool, recipe_name: str, day: int) -> None:
        """Handle cooking/preservation activity."""
        fam = self.family_manager.get_family(villager.family_id)
        source_inv = fam.inventory if fam else villager.personal_inventory
        if source_inv is None:
            return

        recipe = RECIPES.get(recipe_name)
        if recipe is None or not recipe.can_craft(source_inv, 0):
            # Try alternate recipes
            for rname in ("cooked_meat", "bread", "dried_meat", "dried_fish"):
                r = RECIPES.get(rname)
                if r and r.can_craft(source_inv, 0):
                    recipe = r
                    break
            else:
                return

        if success:
            skill = villager.memory.skill_level("cooking", villager.traits.intelligence)
            produced = execute_craft(recipe, source_inv, skill, float(self.rng.random()))
            self.logger.log(
                "ACTIVITY", f"{villager.name} cooked {produced}",
                villager_ids=[villager.id], day=day,
            )

    def _handle_farming(
        self, villager: Villager, activity: str, success: bool,
        tended_positions: set[tuple[int, int]], day: int,
    ) -> None:
        """Handle farming activities."""
        fam = self.family_manager.get_family(villager.family_id)
        if fam is None:
            return

        if activity == "farm_plant" and success:
            # Find available farmland
            farmland = self.resource_manager.get_nearest_of_type(
                villager.current_position, ResourceType.FARMLAND,
            )
            if farmland:
                plot = self.crop_manager.plant(farmland.position, fam.family_id, day)
                fam.farm_plots.append(plot)
                self.logger.log(
                    "ACTIVITY", f"{villager.name} planted crops at {farmland.position}",
                    villager_ids=[villager.id], day=day,
                )

        elif activity == "farm_tend":
            for plot in self.crop_manager.get_family_plots(fam.family_id):
                tended_positions.add(plot.position)
            if success:
                self.logger.log(
                    "ACTIVITY", f"{villager.name} tended crops",
                    villager_ids=[villager.id], day=day,
                )

        elif activity == "farm_harvest" and success:
            harvestable = self.crop_manager.get_harvestable(fam.family_id)
            for plot in harvestable:
                grain_qty = plot.expected_yield
                veg_qty = plot.expected_yield * 0.5
                fam.inventory.add(create_item("grain", grain_qty))
                fam.inventory.add(create_item("vegetables", veg_qty))
                self.crop_manager.remove_harvested(plot)
                self.logger.log(
                    "ACTIVITY",
                    f"{villager.name} harvested {grain_qty:.1f} grain, {veg_qty:.1f} vegetables",
                    villager_ids=[villager.id], day=day,
                )

    def _handle_building(self, villager: Villager, success: bool, day: int) -> None:
        """Handle shelter construction."""
        if not success:
            return
        fam = self.family_manager.get_family(villager.family_id)
        if fam is None:
            return

        existing = self.infrastructure.get_shelter_for(fam.family_id)
        if existing:
            # Improve existing shelter
            self.infrastructure.repair(existing.structure_id, 0.1)
            existing.quality = min(1.0, existing.quality + 0.02)
        else:
            # Build new shelter
            shelter = self.infrastructure.create_structure(
                "shelter", villager.home_position,
                quality=0.3, owner_family_id=fam.family_id,
            )
            fam.shelter_id = shelter.structure_id
            self.logger.log(
                "ACTIVITY", f"{villager.name} built a new shelter",
                villager_ids=[villager.id], day=day,
            )

    def _handle_explore(self, villager: Villager, success: bool, day: int) -> None:
        """Handle exploration — may discover new resource nodes."""
        if success:
            # Discover a random nearby resource
            nodes = self.resource_manager.get_all_in_radius(villager.current_position, 20)
            undiscovered = [n for n in nodes if n.node_id not in villager.memory.known_resource_nodes]
            if undiscovered:
                node = self.rng.choice(undiscovered)
                villager.memory.known_resource_nodes.append(node.node_id)
                self.logger.log(
                    "ACTIVITY",
                    f"{villager.name} discovered {node.resource_type.value} at {node.position}",
                    villager_ids=[villager.id], day=day,
                )

    def _handle_healing(self, villager: Villager, success: bool, day: int) -> None:
        """Handle healing activity."""
        if not success:
            return
        # Heal self or a family member
        fam = self.family_manager.get_family(villager.family_id)
        if fam is None:
            return

        # Find the sickest family member
        target = None
        worst_health = 100
        for vid in fam.member_ids:
            v = self._villager_map.get(vid)
            if v and v.is_alive and v.health < worst_health:
                worst_health = v.health
                target = v

        if target and target.health < 80:
            heal_amount = float(self.rng.uniform(5, 15))
            target.health = min(100, target.health + heal_amount)
            target.needs.satisfy("health", heal_amount / 100)
            # Consume medicine if available
            inv = fam.inventory
            if inv.has("medicine"):
                inv.remove("medicine", 1)
                heal_amount *= 1.5

    # ------------------------------------------------------------------
    # Social phase
    # ------------------------------------------------------------------

    def _resolve_social_phase(self, alive: list[Villager], day: int) -> None:
        """Evening social interactions."""
        from village_sim.core.config import MAX_DAILY_SOCIAL_INTERACTIONS

        for v in alive:
            if v.is_child and v.age_years < 6:
                continue

            n_interactions = int(1 + v.traits.sociability / 100 * (MAX_DAILY_SOCIAL_INTERACTIONS - 1))

            for _ in range(n_interactions):
                action = self.decision_engine.decide_social(
                    v, alive, self.relationship_manager,
                )
                if action is None:
                    break

                target = self._villager_map.get(action.target_id)
                if target is None or not target.is_alive or target.id == v.id:
                    continue

                # Record interaction
                positive = action.action_type in ("chat", "teach", "court", "share_food")
                self.relationship_manager.record_interaction(
                    v.id, target.id, positive, 1.0, day,
                )

                # Satisfy social need
                v.needs.satisfy("social", 0.1)
                target.needs.satisfy("social", 0.05)

                # Knowledge transfer during teaching
                if action.action_type == "teach":
                    self.influence_system.spread_knowledge(
                        v, target, self.relationship_manager, self.rng,
                    )

    # ------------------------------------------------------------------
    # Trade phase
    # ------------------------------------------------------------------

    def _resolve_trade_phase(self, alive: list[Villager], day: int) -> None:
        """Evening trade session: villagers with surplus seek trading partners."""
        from village_sim.core.config import TRADE_MAX_ROUNDS_PER_DAY, TRADE_WILLINGNESS_BASE

        self.trade_system.reset_daily()

        # Shuffle order for fairness
        traders = [v for v in alive if not v.is_child or v.age_years >= 10]
        if len(traders) < 2:
            return

        indices = list(range(len(traders)))
        self.rng.shuffle(indices)

        for idx in indices:
            villager = traders[idx]
            fam = self.family_manager.get_family(villager.family_id)
            if fam is None:
                continue
            villager_inv = fam.inventory

            # Willingness to trade based on personality
            willingness = (
                TRADE_WILLINGNESS_BASE
                + villager.traits.sociability / 100.0 * 0.3
                - (1.0 - villager.traits.risk_tolerance / 100.0) * 0.1
            )
            if self.rng.random() > willingness:
                continue

            # Find trading partners: prefer trusted, then friends, then nearby
            trusted = self.relationship_manager.get_trusted(villager.id, min_trust=0.1)
            friends = self.relationship_manager.get_friends(villager.id, min_affinity=0.2)
            partner_candidates = list(set(trusted + friends))

            # Add some random villagers for market-like encounters
            random_count = max(1, 3 - len(partner_candidates))
            random_others = [v for v in traders if v.id != villager.id and v.id not in partner_candidates]
            if random_others and random_count > 0:
                random_picks = self.rng.choice(
                    random_others,
                    size=min(random_count, len(random_others)),
                    replace=False,
                )
                partner_candidates.extend(random_picks)

            rounds = 0
            for partner_id_or_v in partner_candidates:
                if rounds >= TRADE_MAX_ROUNDS_PER_DAY:
                    break

                if isinstance(partner_id_or_v, int):
                    partner = self._villager_map.get(partner_id_or_v)
                else:
                    partner = partner_id_or_v

                if partner is None or not partner.is_alive or partner.id == villager.id:
                    continue

                partner_fam = self.family_manager.get_family(partner.family_id)
                if partner_fam is None:
                    continue
                partner_inv = partner_fam.inventory

                # Get relationship for trust/familiarity
                rel = self.relationship_manager.get_or_create(villager.id, partner.id)

                # Estimate partner inventory
                partner_estimate = self.trade_system.estimate_partner_inventory(
                    villager, partner, partner_inv, rel.trust, rel.familiarity,
                )

                # Generate offer
                offer = self.trade_system.generate_offer(
                    villager, partner, villager_inv, partner_estimate, rel.trust,
                )
                if offer is None:
                    continue

                # Partner evaluates the offer
                accepted = self.trade_system.evaluate_offer(
                    partner, offer, partner_inv, rel.trust,
                )

                if accepted:
                    success = self.trade_system.execute_trade(
                        offer, villager, partner, villager_inv, partner_inv, day,
                    )
                    if success:
                        # Record positive interaction (trade builds trust)
                        self.relationship_manager.record_interaction(
                            villager.id, partner.id, True, 0.8, day,
                        )
                        items_exchanged = (
                            sum(offer.offering.values()) + sum(offer.requesting.values())
                        )
                        self.metrics.record_trade(items_exchanged)
                        self.logger.log(
                            "TRADE",
                            f"{villager.name} traded {offer.offering} to {partner.name} "
                            f"for {offer.requesting}",
                            villager_ids=[villager.id, partner.id],
                            day=day,
                        )
                else:
                    # Rejected offer — mild negative interaction
                    self.relationship_manager.record_interaction(
                        villager.id, partner.id, False, 0.2, day,
                    )

                rounds += 1

    # ------------------------------------------------------------------
    # Thirst auto-satisfaction
    # ------------------------------------------------------------------

    def _auto_satisfy_thirst(self, alive: list[Villager]) -> None:
        """Auto-satisfy thirst for villagers near water sources."""
        water_nodes = self.resource_manager.get_all_in_radius(
            VILLAGE_CENTER, 100, ResourceType.FRESH_WATER,
        )
        water_positions = set()
        for node in water_nodes:
            # Include positions within radius of water
            for dx in range(-WATER_PROXIMITY_RADIUS, WATER_PROXIMITY_RADIUS + 1):
                for dy in range(-WATER_PROXIMITY_RADIUS, WATER_PROXIMITY_RADIUS + 1):
                    if abs(dx) + abs(dy) <= WATER_PROXIMITY_RADIUS:
                        water_positions.add((node.position[0] + dx, node.position[1] + dy))

        for v in alive:
            if v.current_position in water_positions:
                v.needs.satisfy("thirst", WATER_AUTO_SATISFY_AMOUNT)
            elif v.home_position in water_positions:
                v.needs.satisfy("thirst", WATER_AUTO_SATISFY_AMOUNT * 0.7)

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def _process_lifecycle(self, day: int) -> None:
        """Handle births, deaths, and marriages."""
        alive = [v for v in self.villagers if v.is_alive]

        # Deaths (already marked by daily_update)
        for v in list(self.villagers):
            if not v.is_alive and v not in self.dead_villagers:
                self.dead_villagers.append(v)
                self.metrics.record_death()
                self.logger.log(
                    "LIFECYCLE", f"{v.name} died at age {v.age_years}",
                    villager_ids=[v.id], day=day,
                )
                # Starvation deaths
                if v.needs.needs["hunger"].satisfaction <= 0:
                    self.logger.log("LIFECYCLE", f"{v.name} starved to death", day=day)

        # Births
        for v in alive:
            if v.is_pregnant and v.pregnancy_days >= PREGNANCY_DURATION_DAYS:
                partner_traits = v.traits  # fallback
                if v.spouse_id is not None:
                    spouse = self._villager_map.get(v.spouse_id)
                    if spouse:
                        partner_traits = spouse.traits

                child = v.give_birth(self._next_villager_id, day, partner_traits, self.rng)
                self._next_villager_id += 1
                child.personal_inventory = Inventory(capacity=10.0)
                self.villagers.append(child)
                self._villager_map[child.id] = child

                # Add to family
                fam = self.family_manager.get_family(v.family_id)
                if fam:
                    fam.add_member(child.id)

                self.metrics.record_birth()
                self.logger.log(
                    "LIFECYCLE", f"{child.name} was born to {v.name}",
                    villager_ids=[v.id, child.id], day=day,
                )

        # Marriage proposals
        for v in alive:
            if v.spouse_id is not None or v.age_years < 16:
                continue
            if v.sex != "female":  # either sex can be checked
                continue

            friends = self.relationship_manager.get_friends(v.id, min_affinity=0.5)
            for fid in friends:
                partner = self._villager_map.get(fid)
                if (
                    partner and partner.is_alive
                    and partner.spouse_id is None
                    and partner.sex != v.sex
                    and partner.age_years >= 16
                ):
                    rel = self.relationship_manager.get_or_create(v.id, partner.id)
                    from village_sim.core.config import MARRIAGE_MIN_AFFINITY
                    if rel.affinity >= MARRIAGE_MIN_AFFINITY and self.rng.random() < 0.05:
                        # Marriage!
                        v.spouse_id = partner.id
                        partner.spouse_id = v.id
                        self.family_manager.form_marriage(v, partner)
                        self.metrics.record_marriage()
                        self.logger.log(
                            "MARRIAGE",
                            f"{v.name} and {partner.name} married",
                            villager_ids=[v.id, partner.id], day=day,
                        )
                        break

        # Pregnancy initiation
        for v in alive:
            if v.is_fertile and v.spouse_id is not None and not v.is_pregnant:
                spouse = self._villager_map.get(v.spouse_id)
                if spouse and spouse.is_alive:
                    # Small daily chance of conception
                    if self.rng.random() < 0.005:
                        v.is_pregnant = True
                        v.pregnancy_days = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_world_state(self) -> WorldState:
        """Build the read-only world state for decision-making."""
        family_inventories = {
            fam.family_id: fam.inventory
            for fam in self.family_manager.families.values()
        }
        return WorldState(
            season=self.clock.season,
            weather_modifier=self.climate.outdoor_work_modifier(),
            resource_manager=self.resource_manager,
            world_map=self.world_map,
            family_inventories=family_inventories,
            rng=self.rng,
        )

    def _quick_travel_estimate(self, start: tuple[int, int], end: tuple[int, int]) -> float:
        """Quick travel time estimate without full pathfinding."""
        from village_sim.core.config import BASE_TRAVEL_SPEED
        dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return dist / BASE_TRAVEL_SPEED

    def _distribute_starting_tools(self) -> None:
        """Give starting families basic tools."""
        for fam in self.family_manager.families.values():
            fam.inventory.add(create_item("stone_axe", 1, quality=0.5))
            fam.inventory.add(create_item("stone_knife", 2, quality=0.5))
            fam.inventory.add(create_item("wooden_spear", 1, quality=0.5))
            fam.inventory.add(create_item("fishing_rod", 1, quality=0.4))
            fam.inventory.add(create_item("hoe", 1, quality=0.4))
            fam.inventory.add(create_item("firewood", 5, quality=0.5))

    def _create_starting_shelters(self) -> None:
        """Create basic shelters for all families."""
        cx, cy = VILLAGE_CENTER
        i = 0
        for fam in self.family_manager.families.values():
            # Place shelters in a ring around village center
            angle = (i / max(1, len(self.family_manager.families))) * 6.28
            import math
            sx = cx + int(math.cos(angle) * 5)
            sy = cy + int(math.sin(angle) * 5)
            sx = max(0, min(self.world_map.width - 1, sx))
            sy = max(0, min(self.world_map.height - 1, sy))

            shelter = self.infrastructure.create_structure(
                "shelter", (sx, sy),
                quality=0.4, owner_family_id=fam.family_id,
            )
            fam.shelter_id = shelter.structure_id
            fam.home_position = (sx, sy)

            # Update family members' home positions
            for vid in fam.member_ids:
                v = self._villager_map.get(vid)
                if v:
                    v.home_position = (sx, sy)
                    v.current_position = (sx, sy)
            i += 1

    def _distribute_starting_food(self) -> None:
        """Give families starting food reserves (long-shelf-life items)."""
        for fam in self.family_manager.families.values():
            food_amount = STARTING_FOOD_PER_PERSON * len(fam.member_ids)
            fam.inventory.add(create_item("grain", food_amount * 0.5))
            fam.inventory.add(create_item("dried_meat", food_amount * 0.3))
            fam.inventory.add(create_item("dried_fish", food_amount * 0.2))

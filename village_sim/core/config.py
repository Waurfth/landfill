"""All tunable constants for the village simulation.

Every magic number in the codebase must reference this file.
"""

# =============================================================================
# WORLD
# =============================================================================
MAP_WIDTH: int = 200
MAP_HEIGHT: int = 200
VILLAGE_CENTER: tuple[int, int] = (100, 100)
CELL_SIZE_KM: float = 0.25  # each cell represents 250m

TERRAIN_COSTS: dict[str, float] = {
    "path": 1.0,
    "grassland": 1.2,
    "light_forest": 1.5,
    "dense_forest": 2.5,
    "hills": 2.0,
    "rocky": 3.0,
    "swamp": 3.0,
    "river": 4.0,
    "mountain": 5.0,
}

# =============================================================================
# TIME
# =============================================================================
TICKS_PER_DAY: int = 1
DAYS_PER_SEASON: int = 90
SEASONS: list[str] = ["spring", "summer", "autumn", "winter"]
DAYS_PER_YEAR: int = 360

# Daylight hours by season
DAYLIGHT_HOURS: dict[str, float] = {
    "spring": 13.0,
    "summer": 16.0,
    "autumn": 12.0,
    "winter": 10.0,
}

# =============================================================================
# VILLAGER
# =============================================================================
INITIAL_POPULATION: int = 150
WAKING_HOURS: int = 15
MAX_AGE: int = 75
CHILD_MATURITY_AGE: int = 14
ELDER_DECLINE_AGE: int = 55
FERTILITY_AGE_RANGE: tuple[int, int] = (16, 45)
PREGNANCY_DURATION_DAYS: int = 270
BASE_DAILY_FOOD_NEED: float = 1.5
BASE_DAILY_WATER_NEED: float = 1.0
POST_BIRTH_RECOVERY_DAYS: int = 30

# =============================================================================
# TRAIT RANGES (0-100 scale)
# =============================================================================
MALE_STRENGTH_MEAN: float = 60.0
MALE_STRENGTH_STD: float = 12.0
FEMALE_STRENGTH_MEAN: float = 45.0
FEMALE_STRENGTH_STD: float = 12.0
TRAIT_MEAN: float = 50.0
TRAIT_STD: float = 15.0

# Trait correlations (trait_a, trait_b, correlation)
TRAIT_CORRELATIONS: list[tuple[str, str, float]] = [
    ("patience", "conscientiousness", 0.3),
    ("risk_tolerance", "patience", -0.2),
    ("empathy", "sociability", 0.3),
]

# =============================================================================
# NEEDS - decay rates (per day, proportion of max)
# =============================================================================
HUNGER_DECAY_RATE: float = 0.35
THIRST_DECAY_RATE: float = 0.50
SHELTER_DECAY_RATE: float = 0.05
WARMTH_DECAY_RATE: float = 0.10
SAFETY_DECAY_RATE: float = 0.02
SOCIAL_DECAY_RATE: float = 0.05
REST_DECAY_RATE: float = 0.30
PURPOSE_DECAY_RATE: float = 0.01
COMFORT_DECAY_RATE: float = 0.03

# Need weights (importance in decision-making)
NEED_WEIGHTS: dict[str, float] = {
    "hunger": 10.0,
    "thirst": 12.0,
    "rest": 8.0,
    "warmth": 7.0,
    "shelter": 5.0,
    "safety": 6.0,
    "health": 9.0,
    "social": 3.0,
    "purpose": 2.0,
    "comfort": 1.0,
}

# Survival-critical threshold
SURVIVAL_CRITICAL_THRESHOLD: float = 0.15

# =============================================================================
# SENTIMENT
# =============================================================================
SENTIMENT_CONTAGION_RATE: float = 0.1
SENTIMENT_DECAY_TOWARD_BASELINE: float = 0.02

# =============================================================================
# MOVEMENT
# =============================================================================
BASE_TRAVEL_SPEED: float = 8.0       # cells per hour, flat terrain, no load
CARRY_CAPACITY_BASE: float = 30.0    # kg base, modified by strength
FATIGUE_PER_TRAVEL_HOUR: float = 0.05

# =============================================================================
# RESOURCE REGENERATION
# =============================================================================
FOREST_REGEN_RATE: float = 0.02
FISH_REGEN_RATE: float = 0.08
FARM_YIELD_SEASON: str = "autumn"
MINE_REGEN_RATE: float = 0.0
WILD_PLANTS_REGEN_RATE: float = 0.08
HERB_REGEN_RATE: float = 0.01

# =============================================================================
# SOCIAL
# =============================================================================
RELATIONSHIP_DECAY_RATE: float = 0.001
TRUST_GAIN_PER_POSITIVE: float = 0.05
TRUST_LOSS_PER_NEGATIVE: float = 0.10
MARRIAGE_MIN_AFFINITY: float = 0.6
STATUS_FROM_WEALTH: float = 0.3
STATUS_FROM_SKILL: float = 0.3
STATUS_FROM_SOCIAL: float = 0.2
STATUS_FROM_AGE: float = 0.2
SOCIAL_INFLUENCE_RADIUS: int = 10      # cells
MAX_DAILY_SOCIAL_INTERACTIONS: int = 5
MIN_DAILY_SOCIAL_INTERACTIONS: int = 1

# =============================================================================
# STARTING CONDITIONS
# =============================================================================
STARTING_FOOD_PER_PERSON: float = 45.0  # days of food
STARTING_TOOLS: bool = True
STARTING_SHELTERS: bool = True

# =============================================================================
# SKILLS / LEARNING
# =============================================================================
SKILL_LEARNING_RATE: float = 50.0    # XP units for ~63% skill level
INTELLIGENCE_LEARNING_BONUS: float = 0.5  # max bonus from high intelligence

# =============================================================================
# TOOLS / DURABILITY
# =============================================================================
TOOL_DURABILITY_LOSS_PER_USE: float = 0.5  # per activity session

# =============================================================================
# INVENTORY CAPACITY
# =============================================================================
FAMILY_INVENTORY_CAPACITY: float = 200.0   # kg
COMMUNITY_INVENTORY_CAPACITY: float = 2000.0

# =============================================================================
# DECISION ENGINE
# =============================================================================
HABIT_INERTIA_BONUS: float = 0.05    # bonus for repeating yesterday's activity
SATISFICE_THRESHOLD: float = 0.6     # "good enough" score to stop evaluating
FATIGUE_STOP_THRESHOLD: float = 0.9  # too tired to continue

# =============================================================================
# FARMING / CROPS
# =============================================================================
CROP_GROWTH_DAYS: int = 90            # planting to harvestable
CROP_FAILURE_FROST_THRESHOLD: float = 15.0  # temp below this kills crops
CROP_FAILURE_DROUGHT_DAYS: int = 20   # consecutive no-rain days kills crops

# =============================================================================
# INFRASTRUCTURE
# =============================================================================
SHELTER_CAPACITY_BASE: int = 6        # people per basic shelter
SHELTER_DAILY_DEGRADATION: float = 0.001
STORM_DEGRADATION_MULTIPLIER: float = 5.0

# =============================================================================
# EVENTS
# =============================================================================
STORM_PROBABILITY: float = 0.03
DISEASE_BASE_PROBABILITY: float = 0.005
PREDATOR_PROBABILITY: float = 0.02
PEST_PROBABILITY: float = 0.01
DISCOVERY_PROBABILITY: float = 0.05

# =============================================================================
# THIRST AUTO-SATISFY
# =============================================================================
WATER_PROXIMITY_RADIUS: int = 2         # cells from fresh water
WATER_AUTO_SATISFY_AMOUNT: float = 0.4  # thirst satisfaction when near water

# =============================================================================
# TRADE
# =============================================================================
TRADE_MAX_ROUNDS_PER_DAY: int = 3        # max trade attempts per villager per day
TRADE_SURPLUS_DAYS_THRESHOLD: int = 5    # items beyond N days' needs count as surplus
TRADE_DEFICIT_DAYS_THRESHOLD: int = 3    # items below N days' needs count as deficit
TRADE_WILLINGNESS_BASE: float = 0.5      # base chance a villager wants to trade
TRADE_TRUST_WEIGHT: float = 0.3          # how much trust affects trade willingness
TRADE_PERSONALITY_MARGIN: float = 0.15   # how much personality affects fair-value threshold
TRADE_VALUE_FOOD_HUNGRY_MULTIPLIER: float = 3.0  # how much more food is worth when hungry
TRADE_DIMINISHING_SURPLUS_FACTOR: float = 0.5     # value drops for items already in surplus

# =============================================================================
# DASHBOARD
# =============================================================================
DASHBOARD_UPDATE_INTERVAL: int = 5  # update every N simulated days

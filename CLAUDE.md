# Village Socioeconomic Simulation (village_sim)

Agent-based simulation of a medieval village where economics and society emerge from individual personality-driven behavior. ~6000 lines of Python across 37 modules.

## Running

```bash
# Main simulation (1 year, 150 villagers)
python -m village_sim --days 360 --population 150 --seed 42 --output-dir results

# Quick test run (no dashboard)
python -m village_sim --days 30 --no-dashboard --seed 42

# Monte Carlo analysis (N seeds)
python village_sim/monte_carlo.py --runs 20 --days 90 --output-dir results/monte_carlo

# CLI flags: --days, --population, --seed, --verbosity (0-3), --output-dir, --no-dashboard, --log-file
```

## Architecture

```
village_sim/
  core/       config.py (ALL constants), clock.py (seasons/time)
  world/      map.py (80x80 grid), resources.py (11 types), climate.py, crops.py, infrastructure.py, pathfinding.py (A*)
  agents/     villager.py, personality.py (12 traits, Cholesky correlation), needs.py (10 Maslow needs), memory.py (skills/XP), decision.py (satisficing heuristics)
  economy/    inventory.py (56-item catalog, 3-tier), activities.py (20+ activities), crafting.py (18 recipes), trade.py (bilateral barter, subjective value)
  social/     relationships.py (asymmetric trust), family.py, groups.py (work parties), influence.py (sentiment contagion)
  simulation/ engine.py (14-step daily tick), events.py (storms/disease/predators), metrics.py (Gini, CSV export)
  viz/        dashboard.py (matplotlib 2x4 real-time), logger.py (structured JSON/text)
  monte_carlo.py  # Multi-seed statistical analysis
```

## Key Design Patterns

- **Single RNG stream**: `np.random.default_rng(seed)` in `SimulationEngine.__init__`, passed by reference to ALL subsystems. Same seed = identical output.
- **14-step daily tick** in `engine.py`: dawn (world update) -> decisions -> work parties -> activities -> thirst -> social -> trade -> family food -> need decay -> sentiment contagion -> lifecycle (births/deaths/marriages) -> inventory perish -> infrastructure degrade -> metrics
- **Personality drives everything**: 12 traits (0-100) with Cholesky-correlated generation. Traits modify activity success, decision scoring, social behavior, trade willingness.
- **All constants in `core/config.py`**: ~220 tunable parameters. Never hardcode numbers elsewhere.
- **Emergent behavior**: No global optimizer. Villagers use satisficing heuristics with personality biases.

## Dependencies

- numpy (RNG, trait generation, terrain noise)
- matplotlib (dashboard, static plots)
- No other external dependencies

## Output Files (in --output-dir)

- `metrics.csv` — daily time-series (population, food, sentiment, Gini, trades, skills)
- `events.json` — structured event log
- `simulation.log` — human-readable narrative
- PNG plots: population, food_security, sentiment, inequality, trade_volume, skill_development

## Known Issues / Current State

- **Food economy imbalance**: All 20 Monte Carlo seeds hit 0% hunger by day 55-67. gather_berries dominates (85% success, no tools) while hunting (50%, needs spear) and fishing (60%, needs rod) can't compete. Root causes:
  - River is 1 cell wide -> only ~48 fish nodes
  - Game/fish regeneration rates far too low for 150 people (demand: 225 food/day)
  - Tool durability burns out hunting/fishing capacity
  - Farming takes 150 days to harvest, can't contribute in time
  - Decision engine score = urgency x success_prob, so berries always win
- **Rebalancing needed** in: config.py (regen rates, tool durability), activities.py (success/output tuning), resources.py (regen rates), map.py (river width)

## Spec File

Full design spec at: `C:\Users\ronwa\Downloads\VILLAGE_SIM_SPEC.md` (1,722 lines)

# Village Socioeconomic Simulation (village_sim)

Agent-based simulation of a medieval village where economics and society emerge from individual personality-driven behavior. ~6000 lines of Python across 37 modules.

## Running

```bash
# Easiest: double-click run_simulation.bat (Windows)
# Or from command line:
python run_simulation.py                              # Full year, default settings
python run_simulation.py --days 90 --seed 7           # Custom run

# Module-style (no popup graphs):
python -m village_sim --days 360 --population 150 --seed 42 --output-dir results
python -m village_sim --days 30 --no-dashboard --seed 42

# Monte Carlo analysis (N seeds):
python -c "import sys; sys.path.insert(0,'.'); from village_sim.monte_carlo import monte_carlo; monte_carlo(n_runs=10, days=360)"

# CLI flags: --days, --population, --seed, --verbosity (0-3), --output-dir, --no-dashboard, --log-file
```

**Note**: On this machine, use full Python path: `C:\Users\ronwa\AppData\Local\Python\bin\python3`

## Architecture

```
village_sim/
  core/       config.py (ALL constants), clock.py (seasons/time)
  world/      map.py (200x200 grid), resources.py (11 types), climate.py, crops.py, infrastructure.py, pathfinding.py (A*)
  agents/     villager.py (passive health recovery), personality.py (12 traits, Cholesky), needs.py (10 Maslow needs), memory.py (skills/XP), decision.py (satisficing + depletion feedback)
  economy/    inventory.py (56-item catalog, 3-tier), activities.py (20+ activities), crafting.py (18 recipes), trade.py (bilateral barter, subjective value)
  social/     relationships.py (asymmetric trust), family.py, groups.py (work parties), influence.py (sentiment contagion)
  simulation/ engine.py (14-step daily tick), events.py (storms/disease/predators), metrics.py (Gini, CSV export)
  viz/        dashboard.py (matplotlib 2x4 real-time), logger.py (structured JSON/text)
  monte_carlo.py  # Multi-seed statistical analysis
run_simulation.py   # Standalone runner with popup graphs
run_simulation.bat  # Windows double-click launcher
```

## Key Design Patterns

- **Single RNG stream**: `np.random.default_rng(seed)` in `SimulationEngine.__init__`, passed by reference to ALL subsystems. Same seed = identical output.
- **14-step daily tick** in `engine.py`: dawn (world update) -> decisions -> work parties -> activities -> thirst -> social -> trade -> family food -> need decay -> sentiment contagion -> lifecycle (births/deaths/marriages) -> inventory perish -> infrastructure degrade -> metrics
- **Personality drives everything**: 12 traits (0-100) with Cholesky-correlated generation. Traits modify activity success, decision scoring, social behavior, trade willingness.
- **All constants in `core/config.py`**: ~220 tunable parameters. Never hardcode numbers elsewhere.
- **Emergent behavior**: No global optimizer. Villagers use satisficing heuristics with personality biases.
- **Node randomization**: Villagers pick from nearby resource nodes randomly (not always the nearest) to avoid contention. Execution-time fallback finds alternatives if a planned node is depleted.
- **Resource depletion feedback**: Decision engine scores activities lower when resource nodes are depleted, driving diversification.
- **Passive health recovery**: Well-fed, rested villagers heal 0.5-2.0 HP/day naturally.

## Dependencies

- numpy (RNG, trait generation, terrain noise)
- matplotlib (dashboard, static plots)
- No other external dependencies

## Output Files (in --output-dir)

- `metrics.csv` — daily time-series (population, food, sentiment, Gini, trades, skills)
- `events.json` — structured event log
- `simulation.log` — human-readable narrative
- `summary_dashboard.png` — combined 8-plot dashboard with season shading
- Individual PNGs: population, food_security, sentiment, inequality, trade_volume, skill_development

## Economy Balance (current state)

Village sustains itself through a full year (360 days) with population growth:
- **Seed 42 result**: 150 -> 160 population, 13 births, 3 deaths, 97.4% health, 27.75 food/cap
- Food production is self-sustaining across all seasons including winter
- Activity mix: gathering (dominant), fishing, farming, hunting, healing
- Starting food: 50% grain (180-day shelf), 30% dried meat (60-day), 20% dried fish (90-day)
- Key tuning: berry food_value=0.8, fish perish=5d, meat perish=5d, wild plant regen=0.08, node abundance 80-150

### Economy tuning history
Previous starvation (100% extinction by day 55-67) was caused by:
1. All villagers targeting same resource node (fixed: node randomization + execution fallback)
2. Berry food_value too low (0.5 -> 0.8) and fish perishing too fast (2d -> 5d)
3. Resource nodes too small for 150 villagers (max_abundance 20-40 -> 80-150)
4. No health recovery (added passive recovery for well-fed villagers)
5. Winter seasonal modifiers too harsh (wild plants 0.2 -> 0.4)
6. Tool skill barriers too high (spear skill 5 -> 2, fishing rod 10 -> 5)

## Spec File

Full design spec at: `C:\Users\ronwa\Downloads\VILLAGE_SIM_SPEC.md` (1,722 lines)

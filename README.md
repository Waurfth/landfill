# Village Socioeconomic Simulation

Agent-based simulation of a medieval village where economics and society emerge from individual personality-driven behavior. ~6000 lines of Python across 37 modules.

Villagers have 12 personality traits, 10 Maslow-inspired needs, and make decisions using satisficing heuristics. No global optimizer — all behavior is emergent.

## Quick Start

```bash
# Full year simulation with summary dashboard
python run_simulation.py

# Custom run
python run_simulation.py --days 90 --seed 7 --population 200

# Dwarf Fortress-style grid view (live simulation)
python run_simulation.py --grid --days 360

# Standalone world viewer
python view_world.py                       # Static snapshot
python view_world.py --days 360            # Live simulation
python view_world.py --days 90 --seed 7    # Custom live run

# Windows: double-click run_simulation.bat
```

## Grid View (Dwarf Fortress-style)

An interactive colored-tile world map with real-time simulation overlay.

![Grid View](results/grid_view_default.png)

### Features

- **Terrain rendering** — 200x200 procedurally generated world with 9 terrain types (grassland, light/dense forest, hills, rocky, mountain, river, swamp, paths), colored with elevation shading
- **Villager tracking** — Each villager shown as a colored dot matching their current activity (pink = gathering, yellow = crafting, blue = fishing, red = hunting, etc.) with glow halos for visibility
- **Building overlay** — Shelters (gold), wells (cyan), meeting halls (orange), granaries (brown)
- **Resource overlay** — Toggle to see resource node distribution and abundance levels across the map
- **ASCII mode** — Zoom in (x3+) to see terrain characters on each tile
- **Info sidebar** — Day/season, population, need satisfaction bars, activity breakdown, structure counts, village stockpile totals

### Controls

| Key | Action |
|-----|--------|
| `WASD` / Arrow keys | Pan viewport |
| `Z` / `X` | Zoom in / out |
| `C` | Center on village |
| `R` | Toggle resource node overlay |
| `G` | Toggle grid lines |
| `Space` | Pause / resume simulation |
| `Q` | Quit |

### Launch Options

```bash
python view_world.py --days 360              # Live sim, default speed
python view_world.py --days 360 --speed 10   # Faster animation
python view_world.py --seed 123              # Different world generation
python run_simulation.py --grid --grid-speed 20 --days 180
```

## Architecture

```
village_sim/
  core/       config.py (ALL constants), clock.py (seasons/time)
  world/      map.py (200x200 grid), resources.py (11 types), climate.py,
              crops.py, infrastructure.py, pathfinding.py (A*)
  agents/     villager.py, personality.py (12 traits, Cholesky),
              needs.py (10 Maslow needs), memory.py (3-tier episodic memory),
              decision.py (satisficing + depletion feedback + memory bias)
  economy/    inventory.py (56-item catalog, 3-tier), activities.py (20+),
              crafting.py (18 recipes), trade.py (bilateral barter)
  social/     relationships.py (asymmetric trust), family.py,
              groups.py (work parties), influence.py (sentiment contagion)
  simulation/ engine.py (14-step daily tick), events.py (storms/disease),
              metrics.py (Gini coefficient, CSV export)
  viz/        grid_view.py (DF-style world view), dashboard.py (matplotlib),
              logger.py (structured JSON/text)
  monte_carlo.py
run_simulation.py     # Main runner with popup graphs
view_world.py         # Standalone grid viewer
run_simulation.bat    # Windows double-click launcher
```

## Simulation Design

### Daily Tick (14 steps)

1. Dawn — world update (climate, resource regeneration)
2. Random events (storms, disease, predators)
3. Morning decisions (personality-driven activity planning)
4. Form work parties
5. Execute activities (gathering, hunting, farming, crafting, building...)
6. Auto-satisfy thirst (proximity to water)
7. Evening social interactions
8. Trade phase (bilateral barter with subjective value)
9. Family food distribution + need decay + infrastructure bonuses
10. Sentiment contagion
11. Lifecycle events (births, deaths, marriages)
12. Inventory perishables
13. Infrastructure degradation
14. Metrics collection

### Need System (Maslow-inspired)

10 needs with independent decay rates and urgency curves:

| Need | Decay | Type | Sources |
|------|-------|------|---------|
| Hunger | 0.35/day | Exponential | Food gathering, farming, hunting, fishing |
| Thirst | 0.30/day | Exponential | Proximity to water, wells |
| Rest | 0.30/day | Exponential | Rest activity |
| Warmth | 0.10/day | Exponential | Firewood, shelters |
| Shelter | 0.05/day | Linear | Building shelters |
| Safety | 0.02/day | Linear | Shelters, granaries |
| Health | No decay | Exponential | Healing, medicine |
| Social | 0.05/day | Linear | Socializing, meeting halls |
| Purpose | 0.01/day | Linear | Productive work |
| Comfort | 0.06/day | Linear | Cooking, shelters, crafting |

### Personality Traits (12)

Generated with Cholesky-correlated distributions (0-100 scale):
strength, endurance, dexterity, intelligence, patience, creativity, sociability, empathy, risk_tolerance, conscientiousness, ambition, adaptability

### Episodic Memory (3-tier)

Life events shape future behavior through a three-tier cognitive memory system:

| Tier | Storage | Purpose | Lifetime |
|------|---------|---------|----------|
| **Episodes** | Max 40 detailed events | Salience-weighted sentiment, immediate bias | ~50 days (salience decays at 0.97/day) |
| **Impressions** | ~20 compressed summaries | Experiential trait modifiers | Permanent |
| **Activity Biases** | dict of ±0.3 scores | Learned avoidance/attraction in decisions | Slow decay (0.001/day) |

**How it works:**
- Significant events (injuries, deaths, marriages, births, trades, starvation, festivals) create Episodes
- Episodes decay daily; when salience drops below threshold, they consolidate into Impressions
- Traumatic events (|impact| ≥ 0.7) resist consolidation, persisting longer
- Activity biases spread to related activities (e.g., injury during hunting also biases against similar hunting)
- Accumulated impressions create soft personality drift (grief reduces optimism, injuries reduce risk tolerance, starvation increases loss aversion)
- Social memory: villagers avoid those associated with strongly negative experiences

### Economy

- **20+ activities** mapped to needs — villagers choose activities based on which needs are most urgent
- **Satisficing decisions** — pick the first "good enough" option, biased by personality and memory
- **Tool chains** — axes for woodcutting, knives for cooking, spears for hunting, hoes for farming
- **Toolless fallbacks** — gather_wood and gather_stone break circular tool dependencies
- **Bilateral barter trade** — subjective value based on personal need, trust affects willingness
- **Resource depletion feedback** — depleted nodes score lower, driving activity diversification

## Dependencies

- Python 3.10+
- numpy
- matplotlib

No other external dependencies.

## CLI Reference

```
python run_simulation.py [OPTIONS]

  --days N            Simulation length (default: 360)
  --population N      Initial villagers (default: 150)
  --seed N            RNG seed (default: 42)
  --output-dir DIR    Output directory (default: results/)
  --verbosity {0-3}   Log verbosity
  --grid              Launch DF-style grid view instead of headless sim
  --grid-speed MS     Animation speed in ms/frame (default: 50, lower=faster)

python view_world.py [OPTIONS]

  --days N            Live sim days (0 = static snapshot, default: 0)
  --seed N            RNG seed (default: 42)
  --population N      Initial villagers (default: 150)
  --speed MS          Animation speed (default: 50)
```

## Output Files

Generated in `--output-dir` (default: `results/`):

- `metrics.csv` — daily time-series (population, food, sentiment, Gini, trades, skills)
- `events.json` — structured event log
- `simulation.log` — human-readable narrative
- `summary_dashboard.png` — combined 8-plot dashboard with season shading
- Individual PNGs: population, food_security, sentiment, inequality, trade_volume, skill_development

"""Monte Carlo analysis: run N simulations with different seeds, aggregate statistics."""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    """Summary of a single simulation run."""
    seed: int
    final_population: int
    total_births: int
    total_deaths: int
    total_marriages: int
    total_trades: int
    total_trade_items: float
    final_food_per_capita: float
    final_gini: float
    final_avg_sentiment: float
    final_avg_health: float
    final_avg_wellbeing: float
    final_avg_hunger: float
    peak_population: int
    min_population: int
    starvation_day: int  # first day hunger hit 0, or -1
    dominant_activity: str
    top_skill: str
    top_skill_level: float
    elapsed_seconds: float


def run_single(seed: int, days: int, population: int) -> RunResult:
    """Run one simulation and return summary."""
    from village_sim.simulation.engine import SimulationEngine

    engine = SimulationEngine(seed=seed, population=population)
    engine.logger.verbosity = -1
    engine.logger._stdout = False

    engine.initialize()

    t0 = time.time()
    engine.run(days)
    elapsed = time.time() - t0

    snaps = engine.metrics.snapshots
    last = snaps[-1] if snaps else None

    total_births = sum(s.births for s in snaps)
    total_deaths = sum(s.deaths for s in snaps)
    total_marriages = sum(s.marriage_count for s in snaps)
    total_trades = sum(s.trade_count for s in snaps)
    total_trade_items = sum(s.trade_items_exchanged for s in snaps)
    peak_pop = max(s.population for s in snaps) if snaps else population
    min_pop = min(s.population for s in snaps) if snaps else population

    # First day hunger satisfaction hit 0
    starvation_day = -1
    for s in snaps:
        if s.avg_hunger <= 0.01:
            starvation_day = s.day
            break

    # Dominant activity on final day
    dominant = "idle"
    if last and last.activity_counts:
        dominant = max(last.activity_counts, key=last.activity_counts.get)

    # Top skill
    top_skill = ""
    top_skill_level = 0.0
    if last and last.avg_skill_levels:
        top_skill = max(last.avg_skill_levels, key=last.avg_skill_levels.get)
        top_skill_level = last.avg_skill_levels[top_skill]

    return RunResult(
        seed=seed,
        final_population=last.population if last else 0,
        total_births=total_births,
        total_deaths=total_deaths,
        total_marriages=total_marriages,
        total_trades=total_trades,
        total_trade_items=total_trade_items,
        final_food_per_capita=last.food_per_capita if last else 0,
        final_gini=last.gini if last else 0,
        final_avg_sentiment=last.avg_sentiment if last else 0,
        final_avg_health=last.avg_health if last else 0,
        final_avg_wellbeing=last.avg_wellbeing if last else 0,
        final_avg_hunger=last.avg_hunger if last else 0,
        starvation_day=starvation_day,
        peak_population=peak_pop,
        min_population=min_pop,
        dominant_activity=dominant,
        top_skill=top_skill,
        top_skill_level=top_skill_level,
        elapsed_seconds=elapsed,
    )


def monte_carlo(
    n_runs: int = 20,
    days: int = 90,
    population: int = 150,
    output_dir: str = "results/monte_carlo",
) -> list[RunResult]:
    """Run N simulations with sequential seeds and report aggregate stats."""

    os.makedirs(output_dir, exist_ok=True)
    results: list[RunResult] = []
    rng = np.random.default_rng(0)
    seeds = [int(s) for s in rng.integers(0, 100_000, size=n_runs)]

    print(f"=== Monte Carlo Simulation ===")
    print(f"Runs: {n_runs} | Days/run: {days} | Population: {population}")
    print(f"Seeds: {seeds[:5]}{'...' if n_runs > 5 else ''}")
    print()

    total_t0 = time.time()

    for i, seed in enumerate(seeds):
        t0 = time.time()
        result = run_single(seed, days, population)
        results.append(result)
        elapsed = time.time() - t0
        status = "SURVIVED" if result.final_population > 0 else "EXTINCT"
        print(
            f"  Run {i+1:>3}/{n_runs} | seed={seed:>5} | "
            f"pop {population}->{result.final_population:>3} | "
            f"deaths={result.total_deaths:>3} | "
            f"trades={result.total_trades:>5} | "
            f"food/cap={result.final_food_per_capita:>5.1f} | "
            f"{status} | {elapsed:.1f}s"
        )

    total_elapsed = time.time() - total_t0
    print(f"\nAll {n_runs} runs completed in {total_elapsed:.1f}s "
          f"({total_elapsed/n_runs:.1f}s avg)")

    # ── Aggregate Statistics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    def stat_line(label: str, values: list[float], fmt: str = ".1f") -> str:
        if not values:
            return f"  {label}: no data"
        mn = min(values)
        mx = max(values)
        avg = statistics.mean(values)
        med = statistics.median(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return f"  {label:<30s}  mean={avg:{fmt}}  median={med:{fmt}}  std={std:{fmt}}  min={mn:{fmt}}  max={mx:{fmt}}"

    # Population
    print("\nPOPULATION")
    print(stat_line("Final population", [r.final_population for r in results]))
    print(stat_line("Peak population", [r.peak_population for r in results]))
    print(stat_line("Min population", [r.min_population for r in results]))
    print(stat_line("Total births", [r.total_births for r in results]))
    print(stat_line("Total deaths", [r.total_deaths for r in results]))

    extinct = sum(1 for r in results if r.final_population == 0)
    print(f"  Extinction rate: {extinct}/{n_runs} ({extinct/n_runs*100:.0f}%)")

    # Economy
    print("\nECONOMY")
    print(stat_line("Total trades", [r.total_trades for r in results]))
    print(stat_line("Total items exchanged", [r.total_trade_items for r in results]))
    print(stat_line("Final food/capita", [r.final_food_per_capita for r in results], ".2f"))
    print(stat_line("Final Gini", [r.final_gini for r in results], ".3f"))
    print(stat_line("Total marriages", [r.total_marriages for r in results]))

    # Wellbeing
    print("\nWELLBEING")
    print(stat_line("Final sentiment", [r.final_avg_sentiment for r in results]))
    print(stat_line("Final health", [r.final_avg_health for r in results]))
    print(stat_line("Final wellbeing", [r.final_avg_wellbeing * 100 for r in results]))
    print(stat_line("Final hunger sat.", [r.final_avg_hunger * 100 for r in results]))

    starved = [r for r in results if r.starvation_day >= 0]
    if starved:
        starve_days = [r.starvation_day for r in starved]
        print(f"  Starvation onset: {len(starved)}/{n_runs} runs "
              f"(avg day {statistics.mean(starve_days):.0f}, "
              f"range {min(starve_days)}-{max(starve_days)})")
    else:
        print(f"  Starvation onset: 0/{n_runs} runs")

    # Skills
    print("\nSKILLS")
    skill_freq: dict[str, int] = {}
    for r in results:
        skill_freq[r.top_skill] = skill_freq.get(r.top_skill, 0) + 1
    for skill, count in sorted(skill_freq.items(), key=lambda x: -x[1]):
        print(f"  Top skill '{skill}': {count}/{n_runs} runs "
              f"({count/n_runs*100:.0f}%)")

    # Activities
    print("\nDOMINANT FINAL-DAY ACTIVITY")
    act_freq: dict[str, int] = {}
    for r in results:
        act_freq[r.dominant_activity] = act_freq.get(r.dominant_activity, 0) + 1
    for act, count in sorted(act_freq.items(), key=lambda x: -x[1]):
        print(f"  '{act}': {count}/{n_runs} runs ({count/n_runs*100:.0f}%)")

    # ── Export CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "monte_carlo_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "final_pop", "peak_pop", "min_pop", "births", "deaths",
            "marriages", "trades", "trade_items", "food_per_cap", "gini",
            "sentiment", "health", "wellbeing", "hunger_sat",
            "starvation_day", "dominant_activity", "top_skill",
            "top_skill_level", "elapsed_s",
        ])
        for r in results:
            writer.writerow([
                r.seed, r.final_population, r.peak_population,
                r.min_population, r.total_births, r.total_deaths,
                r.total_marriages, r.total_trades,
                f"{r.total_trade_items:.1f}",
                f"{r.final_food_per_capita:.2f}", f"{r.final_gini:.3f}",
                f"{r.final_avg_sentiment:.1f}", f"{r.final_avg_health:.1f}",
                f"{r.final_avg_wellbeing:.3f}",
                f"{r.final_avg_hunger:.3f}",
                r.starvation_day, r.dominant_activity, r.top_skill,
                f"{r.top_skill_level:.1f}", f"{r.elapsed_seconds:.1f}",
            ])
    print(f"\nResults exported to {csv_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo village simulation")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs")
    parser.add_argument("--days", type=int, default=90, help="Days per run")
    parser.add_argument("--population", type=int, default=150, help="Initial population")
    parser.add_argument("--output-dir", type=str, default="results/monte_carlo")
    args = parser.parse_args()

    monte_carlo(
        n_runs=args.runs,
        days=args.days,
        population=args.population,
        output_dir=args.output_dir,
    )

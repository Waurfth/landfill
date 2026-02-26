"""
Village Simulation Runner
=========================
Run the simulation and display interactive graphs + summary at the end.

Usage:
    python run_simulation.py                          # defaults: 360 days, seed 42
    python run_simulation.py --days 90 --seed 7       # custom run
    python run_simulation.py --help                    # full options

Or double-click run_simulation.bat on Windows.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure village_sim package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Village Socioeconomic Simulation — Run & Visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=360, help="Number of days to simulate")
    parser.add_argument("--population", type=int, default=150, help="Initial population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    from village_sim.simulation.engine import SimulationEngine
    from village_sim.viz.logger import SimLogger

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Village Socioeconomic Simulation")
    print("=" * 60)
    print(f"  Population : {args.population}")
    print(f"  Days       : {args.days}")
    print(f"  Seed       : {args.seed}")
    print(f"  Output     : {args.output_dir}/")
    print("=" * 60)
    print()

    # ── Initialize ──────────────────────────────────────────────────────
    engine = SimulationEngine(seed=args.seed, population=args.population)
    engine.logger = SimLogger(
        verbosity=args.verbosity,
        log_file=os.path.join(args.output_dir, "simulation.log"),
        stdout=(args.verbosity > 0),
    )

    print("Initializing world and population...")
    t0 = time.time()
    engine.initialize()
    print(f"  Done in {time.time() - t0:.2f}s")
    print(f"  World     : {engine.world_map.width}x{engine.world_map.height} grid")
    print(f"  Resources : {len(engine.resource_manager.nodes)} nodes")
    print(f"  Families  : {len(engine.family_manager.families)}")
    print(f"  Shelters  : {len(engine.infrastructure.structures)}")
    print()

    # ── Run simulation with progress ────────────────────────────────────
    print(f"Simulating {args.days} days ...")
    t0 = time.time()
    total = args.days
    milestone = max(1, total // 10)

    try:
        for i in range(total):
            engine.tick()
            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                elapsed = time.time() - t0
                rate = (i + 1) / max(0.01, elapsed)
                pop = sum(1 for v in engine.villagers if v.is_alive)
                print(f"  Day {i + 1:>4}/{total}  ({pct:5.1f}%)  |  Pop: {pop}  |  {rate:.1f} days/s")
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    elapsed = time.time() - t0
    print(f"\nSimulation finished in {elapsed:.1f}s")
    print()

    # ── Export data ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    engine.metrics.export_csv(os.path.join(args.output_dir, "metrics.csv"))
    engine.logger.export_json(os.path.join(args.output_dir, "events.json"))
    engine.logger.close()

    # ── Print summary report ────────────────────────────────────────────
    print(engine.metrics.summary_report())

    # ── Save individual PNGs (archival) ─────────────────────────────────
    try:
        from village_sim.viz.dashboard import Dashboard
        Dashboard.comprehensive_report(engine.metrics, args.output_dir)
    except Exception as e:
        print(f"  (Could not save PNGs: {e})")

    # ── Build interactive summary dashboard ─────────────────────────────
    print("Opening results dashboard ...")
    _show_summary_dashboard(engine.metrics, args.output_dir)


# =====================================================================
# Interactive dashboard — all key plots in one figure with plt.show()
# =====================================================================

def _show_summary_dashboard(metrics, output_dir: str) -> None:
    """Create a combined 2x4 dashboard figure and display it interactively."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    snapshots = metrics.snapshots
    if not snapshots:
        print("  No data to display.")
        return

    days = [s.day for s in snapshots]
    seasons = ["spring", "summer", "autumn", "winter"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Village Simulation Results", fontsize=16, fontweight="bold")

    # ── Add season shading helper ───────────────────────────────────
    def shade_seasons(ax):
        colors = {"spring": "#d4edda", "summer": "#fff3cd", "autumn": "#f8d7da", "winter": "#d1ecf1"}
        max_day = days[-1]
        for i in range(4):
            start = i * 90 + 1
            end = min((i + 1) * 90, max_day)
            if start <= max_day:
                ax.axvspan(start, end, alpha=0.15, color=colors[seasons[i]], zorder=0)

    # ── 1. Population ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(days, [s.population for s in snapshots], "b-", linewidth=2)
    ax.set_title("Population")
    ax.set_ylabel("Villagers")
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 2. Food Security ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(days, [s.food_per_capita for s in snapshots], "g-", linewidth=2)
    ax.axhline(y=2.0, color="r", linestyle="--", alpha=0.5, label="Min threshold")
    ax.set_title("Food per Capita")
    ax.set_ylabel("Food units")
    ax.legend(fontsize=7)
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 3. Health & Wellbeing ───────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(days, [s.avg_health for s in snapshots], "b-", linewidth=1.5, label="Health")
    ax.plot(days, [s.avg_wellbeing * 100 for s in snapshots], "g-", linewidth=1.5, label="Wellbeing")
    ax.plot(days, [s.avg_hunger * 100 for s in snapshots], color="orange", linewidth=1.5, label="Hunger Sat.")
    ax.set_title("Health & Wellbeing")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 4. Trade Volume ─────────────────────────────────────────────
    ax = axes[0, 3]
    ax.plot(days, [s.trade_count for s in snapshots], "c-", linewidth=1.5)
    ax.set_title("Trades per Day")
    ax.set_ylabel("Trades")
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 5. Wealth Inequality ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(days, [s.gini for s in snapshots], "r-", linewidth=2)
    ax.set_title("Wealth Inequality (Gini)")
    ax.set_ylabel("Gini coefficient")
    ax.set_ylim(0, 1)
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 6. Activity Distribution (pie) ──────────────────────────────
    ax = axes[1, 1]
    latest = snapshots[-1]
    if latest.activity_counts:
        sorted_acts = sorted(latest.activity_counts.items(), key=lambda x: -x[1])
        top = sorted_acts[:8]
        if len(sorted_acts) > 8:
            top.append(("other", sum(c for _, c in sorted_acts[8:])))
        labels = [a.replace("_", " ").title() for a, _ in top]
        sizes = [c for _, c in top]
        wedge_colors = plt.cm.Set3.colors[:len(top)]
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8},
               colors=wedge_colors)
    ax.set_title("Activities (Final Day)")

    # ── 7. Sentiment ────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(days, [s.avg_sentiment for s in snapshots], "m-", linewidth=2)
    ax.set_title("Average Sentiment")
    ax.set_ylabel("Sentiment (0-100)")
    ax.set_ylim(0, 100)
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── 8. Skill Development ────────────────────────────────────────
    ax = axes[1, 3]
    all_skills: set[str] = set()
    for s in snapshots:
        all_skills.update(s.avg_skill_levels.keys())
    top_skills = sorted(
        all_skills,
        key=lambda sk: snapshots[-1].avg_skill_levels.get(sk, 0),
        reverse=True,
    )[:8]
    for skill_name in top_skills:
        values = [s.avg_skill_levels.get(skill_name, 0) for s in snapshots]
        if max(values) > 0.5:
            ax.plot(days, values, linewidth=1.5, label=skill_name)
    ax.set_title("Skill Development")
    ax.set_ylabel("Avg skill level")
    if top_skills:
        ax.legend(fontsize=7, loc="upper left")
    shade_seasons(ax)
    ax.grid(True, alpha=0.3)

    # ── Layout & save ───────────────────────────────────────────────
    for ax in axes[1, :]:
        ax.set_xlabel("Day")
    for ax in axes[0, :]:
        ax.set_xlabel("Day")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save combined dashboard
    dashboard_path = os.path.join(output_dir, "summary_dashboard.png")
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    print(f"  Dashboard saved to {dashboard_path}")

    # Show interactively (blocks until user closes the window)
    print("  Close the graph window to exit.")
    plt.show()


if __name__ == "__main__":
    run()

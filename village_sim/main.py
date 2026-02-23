"""Entry point for the village socioeconomic simulation."""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Village Socioeconomic Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=360, help="Number of days to simulate")
    parser.add_argument("--population", type=int, default=150, help="Initial population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2, 3], help="Log verbosity level")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable real-time dashboard")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")

    args = parser.parse_args()

    # Import here to allow --help without loading everything
    from village_sim.simulation.engine import SimulationEngine
    from village_sim.viz.logger import SimLogger

    print(f"=== Village Socioeconomic Simulation ===")
    print(f"Population: {args.population} | Days: {args.days} | Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()

    # Create engine
    engine = SimulationEngine(seed=args.seed, population=args.population)

    # Configure logger
    engine.logger = SimLogger(
        verbosity=args.verbosity,
        log_file=args.log_file or os.path.join(args.output_dir, "simulation.log"),
        stdout=(args.verbosity > 0),
    )

    # Initialize
    print("Initializing world and population...")
    t0 = time.time()
    engine.initialize()
    print(f"Initialization complete in {time.time() - t0:.2f}s")
    print(f"  World: {engine.world_map.width}x{engine.world_map.height} grid")
    print(f"  Resources: {len(engine.resource_manager.nodes)} nodes")
    print(f"  Families: {len(engine.family_manager.families)}")
    print(f"  Shelters: {len(engine.infrastructure.structures)}")
    print()

    # Set up dashboard
    dashboard = None
    if not args.no_dashboard:
        try:
            from village_sim.viz.dashboard import Dashboard
            dashboard = Dashboard()
            dashboard.initialize()
            engine.set_dashboard_callback(lambda day, metrics: dashboard.update(day, metrics))
            print("Real-time dashboard enabled")
        except Exception as e:
            print(f"Dashboard unavailable ({e}), continuing without visualization")
            dashboard = None

    # Run simulation
    print(f"Running simulation for {args.days} days...")
    t0 = time.time()

    try:
        engine.run(args.days)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    elapsed = time.time() - t0
    days_run = engine.clock.day
    print(f"\nSimulation complete: {days_run} days in {elapsed:.2f}s ({days_run / max(0.01, elapsed):.0f} days/sec)")

    # Export results
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    engine.metrics.export_csv(csv_path)
    print(f"Metrics exported to {csv_path}")

    # Generate static reports
    try:
        from village_sim.viz.dashboard import Dashboard as DashClass
        DashClass.comprehensive_report(engine.metrics, args.output_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    # Print summary
    print()
    print(engine.metrics.summary_report())

    # Save dashboard
    if dashboard:
        dashboard.save(os.path.join(args.output_dir, "dashboard_final.png"))
        dashboard.close()

    # Export log
    engine.logger.export_json(os.path.join(args.output_dir, "events.json"))
    engine.logger.close()

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

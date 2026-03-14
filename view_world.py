"""
Standalone Village World Viewer
================================
Quick-launch the Dwarf Fortress-style grid visualization.

Usage:
    python view_world.py                    # Static snapshot of generated world
    python view_world.py --days 360         # Live simulation with grid view
    python view_world.py --days 90 --seed 7 # Custom live run
    python view_world.py --speed 10         # Faster animation

Controls:
    WASD / Arrow keys  — Pan viewport
    Z / X              — Zoom in / out
    C                  — Center on village
    R                  — Toggle resource node overlay
    G                  — Toggle grid lines
    Space              — Pause / resume (live mode only)
    Q                  — Quit
"""

from __future__ import annotations

import os
import sys

# Ensure village_sim is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from village_sim.viz.grid_view import launch_grid_view

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Village World Viewer — Dwarf Fortress-style grid visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--population", type=int, default=150, help="Initial population (default: 150)")
    parser.add_argument("--days", type=int, default=0,
                        help="Run live simulation for N days (0 = static snapshot)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Animation speed in ms/frame (lower = faster, default: 50)")
    args = parser.parse_args()

    launch_grid_view(
        seed=args.seed,
        population=args.population,
        days=args.days,
        speed=args.speed,
    )

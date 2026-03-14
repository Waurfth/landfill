"""Dwarf Fortress-style grid visualization for the village simulation.

Interactive colored-tile map with entity overlays and a real-time
information sidebar. Supports panning, zooming, and live simulation.

Controls:
    WASD / Arrow keys  — Pan viewport
    Z / X              — Zoom in / out
    C                  — Center on village
    R                  — Toggle resource node overlay
    G                  — Toggle grid lines
    Space              — Pause / resume (live mode only)
    Q                  — Quit

Usage:
    # Live simulation with grid view:
    python run_simulation.py --grid --days 360

    # Standalone snapshot of a generated world:
    python -m village_sim.viz.grid_view
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from village_sim.core.config import (
    DASHBOARD_UPDATE_INTERVAL,
    MAP_HEIGHT,
    MAP_WIDTH,
    VILLAGE_CENTER,
)

if TYPE_CHECKING:
    from village_sim.simulation.engine import SimulationEngine

# ═══════════════════════════════════════════════════════════════════════
# Color palette — dark theme inspired by Dwarf Fortress
# ═══════════════════════════════════════════════════════════════════════

BG_COLOR = "#0f0f23"
PANEL_BG = "#16213e"
GOLD = "#ffd700"
TEXT = "#c0c0c0"
DIM = "#666688"

# Terrain background colors (R, G, B) — rich and saturated on dark base
TERRAIN_COLORS: dict[str, tuple[float, float, float]] = {
    "grassland":    (0.22, 0.48, 0.18),
    "light_forest": (0.15, 0.40, 0.12),
    "dense_forest": (0.07, 0.24, 0.07),
    "hills":        (0.56, 0.44, 0.24),
    "rocky":        (0.50, 0.50, 0.52),
    "mountain":     (0.66, 0.66, 0.70),
    "swamp":        (0.30, 0.36, 0.20),
    "river":        (0.12, 0.30, 0.62),
    "path":         (0.60, 0.50, 0.32),
}

# Terrain ASCII characters for close-up zoom
TERRAIN_CHARS: dict[str, str] = {
    "grassland":    ".",
    "light_forest": "t",
    "dense_forest": "T",
    "hills":        "n",
    "rocky":        "%",
    "mountain":     "^",
    "swamp":        "~",
    "river":        "=",
    "path":         "#",
}

TERRAIN_CHAR_COLORS: dict[str, str] = {
    "grassland":    "#5a8a44",
    "light_forest": "#44cc33",
    "dense_forest": "#22aa22",
    "hills":        "#ccaa55",
    "rocky":        "#aaaaaa",
    "mountain":     "#eeeeee",
    "swamp":        "#88aa44",
    "river":        "#44aaff",
    "path":         "#ccaa77",
}

# Structure display settings: (marker, color, marker_size)
STRUCTURE_STYLE: dict[str, tuple[str, str, float]] = {
    "shelter":       ("s", "#FFD700",  5),
    "well":          ("D", "#00FFFF", 10),
    "meeting_hall":  ("H", "#FF8C00", 12),
    "granary":       ("8", "#CD853F", 12),
    "bridge":        ("_", "#B0B0B0",  8),
    "storage_shed":  ("s", "#DEB887",  6),
    "road_segment":  (".", "#A0A0A0",  3),
    "farm_plot":     ("+", "#9ACD32",  6),
}

# Resource node colors for overlay dots
RESOURCE_COLORS: dict[str, str] = {
    "timber":          "#338822",
    "game_small":      "#cc4444",
    "game_large":      "#991111",
    "fish":            "#3399ff",
    "stone":           "#999999",
    "clay":            "#aa7744",
    "iron_ore":        "#555577",
    "wild_plants":     "#66cc44",
    "medicinal_herbs": "#cc66ff",
    "farmland":        "#ccaa44",
    "fresh_water":     "#55bbff",
}

# Activity -> villager dot color
ACTIVITY_COLORS: dict[str, str] = {
    "gather_berries":  "#ff66cc",
    "hunt_small_game": "#ff4444",
    "hunt_large_game": "#cc0000",
    "fishing":         "#4488ff",
    "chop_wood":       "#aa6633",
    "gather_wood":     "#886644",
    "gather_stone":    "#aaaaaa",
    "mine_stone":      "#888888",
    "mine_ore":        "#666688",
    "farm_plant":      "#66aa22",
    "farm_tend":       "#88cc44",
    "farm_harvest":    "#aaee44",
    "cook_food":       "#ff8833",
    "preserve_food":   "#cc7722",
    "craft_tools":     "#ffee44",
    "build_shelter":   "#ffcc00",
    "build_well":      "#00cccc",
    "build_meeting_hall": "#ff8800",
    "build_granary":   "#cc8844",
    "gather_herbs":    "#cc66ff",
    "heal_villager":   "#66ff66",
    "rest":            "#555555",
    "socialize":       "#ff99cc",
    "explore":         "#44ffcc",
}

# Need bar colors
NEED_COLORS: dict[str, str] = {
    "hunger":  "#ff9944",
    "thirst":  "#44aaff",
    "rest":    "#9999ee",
    "warmth":  "#ff5533",
    "shelter": "#cc9933",
    "safety":  "#44ee44",
    "health":  "#ff66cc",
    "social":  "#eeee44",
    "purpose": "#aa88ff",
    "comfort": "#ffaa88",
}

SEASON_SYMBOLS: dict[str, str] = {
    "spring": "SPR",
    "summer": "SUM",
    "autumn": "AUT",
    "winter": "WIN",
}


# ═══════════════════════════════════════════════════════════════════════
# Helper: Unicode need bar
# ═══════════════════════════════════════════════════════════════════════

def _need_bar(value: float, width: int = 10) -> str:
    """Render a value (0-1) as a filled/empty block bar."""
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


# ═══════════════════════════════════════════════════════════════════════
# Main visualization class
# ═══════════════════════════════════════════════════════════════════════

class WorldGridView:
    """Interactive Dwarf Fortress-style world grid visualization.

    Renders the 200x200 tile world as a colored grid with entity overlays
    and a real-time information sidebar.
    """

    def __init__(
        self,
        engine: "SimulationEngine",
        viewport_w: int = 80,
        viewport_h: int = 50,
    ) -> None:
        self.engine = engine

        # Viewport state
        self.vw = viewport_w
        self.vh = viewport_h
        cx, cy = VILLAGE_CENTER
        self.vx = max(0, cx - self.vw // 2)
        self.vy = max(0, cy - self.vh // 2)

        # Display toggles
        self.show_resources = False
        self.show_villagers = True
        self.show_grid = False
        self._paused = False
        self._quit = False

        # Zoom level (1=default, 2=close, 3=very close with ASCII chars)
        self.zoom = 1

        # Caches
        self._terrain_rgb: Optional[np.ndarray] = None
        self._fig: Optional[plt.Figure] = None
        self._ax_map: Optional[plt.Axes] = None
        self._ax_info: Optional[plt.Axes] = None

    # ─────────────────────────────────────────────────────────────
    # Terrain image construction (built once, cached)
    # ─────────────────────────────────────────────────────────────

    def _build_terrain_rgb(self) -> np.ndarray:
        """Build a 200x200x3 RGB image from terrain types + elevation."""
        wm = self.engine.world_map
        h, w = wm.height, wm.width

        # Build lookup: terrain_name -> index, and color palette array
        terrain_names = list(TERRAIN_COLORS.keys())
        name_to_idx = {n: i for i, n in enumerate(terrain_names)}
        palette = np.array([TERRAIN_COLORS[n] for n in terrain_names], dtype=np.float32)

        # Collect terrain indices and elevations
        idx_grid = np.zeros((h, w), dtype=np.int32)
        elev_grid = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            row = wm.grid[y]
            for x in range(w):
                cell = row[x]
                if cell.has_bridge:
                    tt = "path"
                elif cell.has_road and cell.terrain_type != "river":
                    tt = "path"
                else:
                    tt = cell.terrain_type
                idx_grid[y, x] = name_to_idx.get(tt, 0)
                elev_grid[y, x] = cell.elevation

        # Vectorized color lookup + elevation shading
        img = palette[idx_grid]  # shape (h, w, 3)
        shade = (0.72 + 0.56 * elev_grid)[:, :, np.newaxis]
        img = np.clip(img * shade, 0.0, 1.0)

        return img

    # ─────────────────────────────────────────────────────────────
    # Figure initialization
    # ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the matplotlib figure, axes, and caches."""
        plt.style.use("dark_background")

        self._fig = plt.figure(
            figsize=(19, 10),
            facecolor=BG_COLOR,
            num="Village Simulation — World Grid",
        )

        # Map axes (left 65%)
        self._ax_map = self._fig.add_axes([0.01, 0.04, 0.63, 0.92])
        self._ax_map.set_facecolor(BG_COLOR)

        # Info panel (right 33%)
        self._ax_info = self._fig.add_axes([0.66, 0.04, 0.33, 0.92])
        self._ax_info.set_facecolor(PANEL_BG)
        self._ax_info.set_xlim(0, 1)
        self._ax_info.set_ylim(0, 1)
        self._ax_info.set_xticks([])
        self._ax_info.set_yticks([])
        for spine in self._ax_info.spines.values():
            spine.set_color("#333366")

        # Build terrain cache
        self._terrain_rgb = self._build_terrain_rgb()

        # Keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ─────────────────────────────────────────────────────────────
    # Keyboard handler
    # ─────────────────────────────────────────────────────────────

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", "")
        step = max(1, 5 // self.zoom)
        wm = self.engine.world_map

        if key in ("w", "up"):
            self.vy = max(0, self.vy - step)
        elif key in ("s", "down"):
            self.vy = min(wm.height - self.vh, self.vy + step)
        elif key in ("a", "left"):
            self.vx = max(0, self.vx - step)
        elif key in ("d", "right"):
            self.vx = min(wm.width - self.vw, self.vx + step)
        elif key == "z":  # zoom in
            if self.zoom < 4:
                self.zoom += 1
                self.vw = max(16, 80 // self.zoom)
                self.vh = max(12, 50 // self.zoom)
                # Re-center
                cx = self.vx + self.vw
                cy = self.vy + self.vh
                self.vx = max(0, min(wm.width - self.vw, cx - self.vw // 2))
                self.vy = max(0, min(wm.height - self.vh, cy - self.vh // 2))
        elif key == "x":  # zoom out
            if self.zoom > 1:
                cx = self.vx + self.vw // 2
                cy = self.vy + self.vh // 2
                self.zoom -= 1
                self.vw = min(wm.width, 80 // self.zoom)
                self.vh = min(wm.height, 50 // self.zoom)
                self.vx = max(0, min(wm.width - self.vw, cx - self.vw // 2))
                self.vy = max(0, min(wm.height - self.vh, cy - self.vh // 2))
        elif key == "c":  # center on village
            cx, cy = VILLAGE_CENTER
            self.vx = max(0, min(wm.width - self.vw, cx - self.vw // 2))
            self.vy = max(0, min(wm.height - self.vh, cy - self.vh // 2))
        elif key == "r":
            self.show_resources = not self.show_resources
        elif key == "g":
            self.show_grid = not self.show_grid
        elif key == " ":
            self._paused = not self._paused
        elif key == "q":
            self._quit = True
            plt.close(self._fig)
            return
        else:
            return

        self.render()

    # ─────────────────────────────────────────────────────────────
    # Main render
    # ─────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Redraw the map and info panel for the current viewport."""
        if self._fig is None or self._terrain_rgb is None:
            return

        self._render_map()
        self._render_info_panel()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _render_map(self) -> None:
        """Render the terrain grid with entity overlays."""
        ax = self._ax_map
        ax.clear()
        ax.set_facecolor(BG_COLOR)

        wm = self.engine.world_map

        # Clamp viewport
        vx = max(0, min(wm.width - self.vw, self.vx))
        vy = max(0, min(wm.height - self.vh, self.vy))
        vw, vh = self.vw, self.vh

        # ── Terrain layer ──
        viewport_img = self._terrain_rgb[vy : vy + vh, vx : vx + vw].copy()

        # Show the image (y-axis: top of viewport at top of display)
        ax.imshow(
            viewport_img,
            interpolation="nearest",
            aspect="equal",
            extent=[vx, vx + vw, vy + vh, vy],
        )

        # ── Resource overlay ──
        if self.show_resources:
            for node in self.engine.resource_manager.nodes:
                nx, ny = node.position
                if vx <= nx < vx + vw and vy <= ny < vy + vh:
                    if node.current_abundance > 0:
                        rcolor = RESOURCE_COLORS.get(node.resource_type.value, "#ffffff")
                        # Scale marker by relative abundance
                        rel = node.current_abundance / max(1, node.max_abundance)
                        ms = 1.5 + 3.0 * rel
                        ax.plot(
                            nx + 0.5, ny + 0.5,
                            "o", color=rcolor, markersize=ms / self.zoom + 0.5,
                            alpha=0.6, markeredgewidth=0,
                        )

        # ── Structure overlay ──
        for struct in self.engine.infrastructure.structures:
            sx, sy = struct.position
            if vx <= sx < vx + vw and vy <= sy < vy + vh:
                marker, color, base_size = STRUCTURE_STYLE.get(
                    struct.structure_type, ("?", "#ffffff", 5)
                )
                ms = base_size / self.zoom + 2
                ax.plot(
                    sx + 0.5, sy + 0.5,
                    marker="s", color=color, markersize=ms,
                    markeredgecolor="#000000", markeredgewidth=0.5,
                )

        # ── Villager overlay ──
        if self.show_villagers:
            alive = [v for v in self.engine.villagers if v.is_alive]
            # Collect positions for batch plotting
            vx_positions: dict[str, list[float]] = {}
            vy_positions: dict[str, list[float]] = {}

            for v in alive:
                px, py = v.current_position
                if vx <= px < vx + vw and vy <= py < vy + vh:
                    act = getattr(v, "current_activity", "rest") or "rest"
                    color = ACTIVITY_COLORS.get(act, "#ffffff")
                    if color not in vx_positions:
                        vx_positions[color] = []
                        vy_positions[color] = []
                    vx_positions[color].append(px + 0.5)
                    vy_positions[color].append(py + 0.5)

            for color, xs in vx_positions.items():
                ys = vy_positions[color]
                ms = 5.0 / self.zoom + 1.5
                # Glow layer (slightly larger, translucent white)
                ax.scatter(
                    xs, ys,
                    s=(ms + 2) ** 2, c="#ffffff", marker="o",
                    edgecolors="none", linewidths=0,
                    zorder=9, alpha=0.15,
                )
                # Main dot
                ax.scatter(
                    xs, ys,
                    s=ms ** 2, c=color, marker="o",
                    edgecolors="#000000", linewidths=0.4,
                    zorder=10, alpha=0.92,
                )

        # ── Zoomed-in ASCII overlay ──
        if self.zoom >= 3:
            for dy in range(vh):
                for dx in range(vw):
                    wx, wy = vx + dx, vy + dy
                    if 0 <= wx < wm.width and 0 <= wy < wm.height:
                        cell = wm.grid[wy][wx]
                        tt = cell.terrain_type
                        if cell.has_road and tt != "river":
                            tt = "path"
                        ch = TERRAIN_CHARS.get(tt, "?")
                        fc = TERRAIN_CHAR_COLORS.get(tt, "#888888")
                        ax.text(
                            wx + 0.5, wy + 0.5, ch,
                            color=fc, fontsize=7 * self.zoom / 3,
                            ha="center", va="center",
                            fontfamily="monospace", fontweight="bold",
                            alpha=0.7,
                        )

        # ── Grid lines ──
        if self.show_grid:
            for gx in range(vx, vx + vw + 1):
                ax.axvline(gx, color="#333355", linewidth=0.3, alpha=0.5)
            for gy in range(vy, vy + vh + 1):
                ax.axhline(gy, color="#333355", linewidth=0.3, alpha=0.5)

        # ── Village center marker ──
        vcx, vcy = VILLAGE_CENTER
        if vx <= vcx < vx + vw and vy <= vcy < vy + vh:
            ax.plot(
                vcx + 0.5, vcy + 0.5,
                marker="+", color="#ffd700", markersize=12 / self.zoom + 3,
                markeredgewidth=2, zorder=20,
            )

        # ── Axes cosmetics ──
        ax.set_xlim(vx, vx + vw)
        ax.set_ylim(vy + vh, vy)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#333366")

        # Title
        day = self.engine.clock.day
        season = self.engine.clock.season
        ssym = SEASON_SYMBOLS.get(season, "???")
        zoom_label = f"x{self.zoom}" if self.zoom > 1 else ""
        pause_label = " [PAUSED]" if self._paused else ""
        ax.set_title(
            f" Village World \u2014 Day {day}  {ssym} {season.capitalize()}"
            f"  {zoom_label}{pause_label}",
            color=GOLD, fontsize=13, fontweight="bold", loc="left",
            fontfamily="monospace",
        )

    # ─────────────────────────────────────────────────────────────
    # Info panel
    # ─────────────────────────────────────────────────────────────

    def _render_info_panel(self) -> None:
        """Render the right-side information panel with DF-style text."""
        ax = self._ax_info
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#333366")

        alive = [v for v in self.engine.villagers if v.is_alive]
        day = self.engine.clock.day
        season = self.engine.clock.season
        pop = len(alive)

        y = 0.97
        dy = 0.022
        fs_header = 10
        fs_data = 8.5
        mono = "monospace"

        def header(text: str) -> None:
            nonlocal y
            ax.text(0.5, y, f"\u2550\u2550\u2550 {text} \u2550\u2550\u2550",
                    color=GOLD, fontsize=fs_header, ha="center",
                    fontweight="bold", fontfamily=mono)
            y -= dy * 1.3

        def line(text: str, color: str = TEXT) -> None:
            nonlocal y
            ax.text(0.04, y, text, color=color, fontsize=fs_data,
                    fontfamily=mono, va="center")
            y -= dy

        def gap(n: float = 0.5) -> None:
            nonlocal y
            y -= dy * n

        # ── Header ──
        header("STATUS")
        ssym = SEASON_SYMBOLS.get(season, "???")
        line(f" Day {day:>4}  {ssym} {season.capitalize()}", "#87ceeb")
        gap()

        # ── Population ──
        header("POPULATION")
        n_dead = len(self.engine.dead_villagers)
        n_children = sum(1 for v in alive if v.is_child)
        n_families = len(self.engine.family_manager.families)
        line(f"  Alive:     {pop:>4}")
        line(f"  Deaths:    {n_dead:>4}", "#ff6666" if n_dead > 0 else TEXT)
        line(f"  Children:  {n_children:>4}")
        line(f"  Families:  {n_families:>4}")
        gap()

        # ── Needs ──
        header("NEEDS")
        need_order = [
            "hunger", "thirst", "rest", "warmth", "shelter",
            "safety", "health", "social", "purpose", "comfort",
        ]
        for need_name in need_order:
            if alive:
                avg = float(np.mean([
                    v.needs.needs[need_name].satisfaction for v in alive
                ]))
            else:
                avg = 0.0
            bar = _need_bar(avg, 10)
            ncolor = NEED_COLORS.get(need_name, TEXT)
            label = f"  {need_name[:7]:>7}"
            pct = f"{avg * 100:>3.0f}%"
            ax.text(0.04, y, label, color=ncolor, fontsize=fs_data,
                    fontfamily=mono, va="center")
            ax.text(0.30, y, bar, color=ncolor, fontsize=fs_data - 1,
                    fontfamily=mono, va="center")
            ax.text(0.82, y, pct, color=ncolor, fontsize=fs_data,
                    fontfamily=mono, va="center")
            y -= dy
        gap()

        # ── Activities ──
        header("ACTIVITIES")
        act_counts: dict[str, int] = {}
        for v in alive:
            act = getattr(v, "current_activity", "rest") or "rest"
            act_counts[act] = act_counts.get(act, 0) + 1
        sorted_acts = sorted(act_counts.items(), key=lambda x: -x[1])

        for act_name, count in sorted_acts[:8]:
            pct = count / max(1, pop) * 100
            acolor = ACTIVITY_COLORS.get(act_name, TEXT)
            line(
                f"  {act_name[:15]:<15} {count:>3} {pct:>4.0f}%",
                acolor,
            )
        if len(sorted_acts) > 8:
            other = sum(c for _, c in sorted_acts[8:])
            line(f"  {'(other)':<15} {other:>3}", DIM)
        gap()

        # ── Structures ──
        header("STRUCTURES")
        struct_counts: dict[str, int] = {}
        for s in self.engine.infrastructure.structures:
            struct_counts[s.structure_type] = struct_counts.get(s.structure_type, 0) + 1
        for stype in ["shelter", "well", "meeting_hall", "granary", "bridge"]:
            cnt = struct_counts.get(stype, 0)
            scolor = STRUCTURE_STYLE.get(stype, ("?", TEXT, 5))[1]
            label = stype.replace("_", " ").title()
            line(f"  {label:<14} {cnt:>3}", scolor if cnt > 0 else DIM)
        gap()

        # ── Resources (stockpile summary) ──
        header("STOCKPILES")
        # Aggregate family inventories
        total_food = 0.0
        total_firewood = 0.0
        total_timber = 0.0
        total_stone = 0.0
        for fam in self.engine.family_manager.families.values():
            total_food += fam.inventory.total_food_value()
            total_firewood += fam.inventory.total_of("firewood")
            total_timber += fam.inventory.total_of("timber")
            total_stone += fam.inventory.total_of("stone")
        line(f"  Food:      {total_food:>7.0f}")
        line(f"  Firewood:  {total_firewood:>7.0f}")
        line(f"  Timber:    {total_timber:>7.0f}")
        line(f"  Stone:     {total_stone:>7.0f}")
        gap()

        # ── Controls ──
        y = max(y, 0.01)
        ax.text(
            0.5, 0.015,
            "WASD:Pan  Z/X:Zoom  C:Center  R:Rsrc  G:Grid",
            color=DIM, fontsize=7, ha="center", fontfamily=mono,
        )

    # ─────────────────────────────────────────────────────────────
    # Public API: show, update, run
    # ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Display the current world state interactively (blocking)."""
        self.initialize()
        self.render()
        plt.show()

    def update(self, day: int, metrics: object) -> None:
        """Callback for SimulationEngine.run() — update display."""
        if self._fig is None:
            self.initialize()
        if day % DASHBOARD_UPDATE_INTERVAL == 0:
            self.render()

    def run_live(
        self,
        days: int = 360,
        update_every: int = 1,
        speed: int = 50,
    ) -> None:
        """Run the simulation with real-time grid visualization.

        Parameters
        ----------
        days : int
            Total simulation days to run.
        update_every : int
            Render every N ticks.
        speed : int
            Milliseconds between animation frames (lower = faster).
        """
        self.initialize()
        self.render()

        self._day_counter = 0
        self._max_days = days
        self._update_every = update_every

        def _step(frame: int) -> None:
            if self._quit:
                return
            if self._paused:
                return
            if self._day_counter >= self._max_days:
                return

            # Run one tick
            self.engine.tick()
            self._day_counter += 1

            # Render
            if self._day_counter % self._update_every == 0:
                self.render()

        ani = FuncAnimation(
            self._fig,
            _step,
            frames=days,
            interval=speed,
            repeat=False,
            cache_frame_data=False,
        )
        # Keep reference to prevent garbage collection
        self._animation = ani
        plt.show()

    def save(self, filepath: str) -> None:
        """Save the current grid view as an image."""
        if self._fig is None:
            self.initialize()
            self.render()
        self._fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)

    def close(self) -> None:
        """Close the visualization window."""
        if self._fig:
            plt.close(self._fig)


# ═══════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════

def launch_grid_view(
    seed: int = 42,
    population: int = 150,
    days: int = 0,
    speed: int = 50,
) -> None:
    """Launch the grid view, optionally running a live simulation.

    Parameters
    ----------
    seed : int
        RNG seed for the simulation.
    population : int
        Initial village population.
    days : int
        If > 0, run a live simulation for this many days.
        If 0, just show the initial world state.
    speed : int
        Milliseconds per animation frame (lower = faster).
    """
    from village_sim.simulation.engine import SimulationEngine

    engine = SimulationEngine(seed=seed, population=population)
    engine.initialize()

    view = WorldGridView(engine)

    if days > 0:
        print(f"Launching grid view with live simulation ({days} days, seed={seed})")
        view.run_live(days=days, speed=speed)
    else:
        print(f"Launching grid view (snapshot, seed={seed})")
        view.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Village world grid viewer")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--population", type=int, default=150, help="Initial population")
    parser.add_argument("--days", type=int, default=0,
                        help="Run live simulation for N days (0 = snapshot only)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Animation speed in ms per frame (lower = faster)")
    args = parser.parse_args()

    launch_grid_view(
        seed=args.seed,
        population=args.population,
        days=args.days,
        speed=args.speed,
    )

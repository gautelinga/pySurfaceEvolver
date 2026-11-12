"""Run the cells of docs/hele_shaw_defects.md and draw its pictures. Needs
matplotlib, PyVista (off-screen rendering) and imageio-ffmpeg (the movie).

    OMP_NUM_THREADS=2 python tools/hele_shaw_figure.py run NAME CENTRE ... [--out OUT]
    python tools/hele_shaw_figure.py draw [--out OUT] [--film NAME]

Parts:

  run    one cell, defects at x = CENTRE ... (0: one defect, in the mirror
         plane; 6.5: two, 13 apart), the paper's cycle: oil imbibed from a
         flat meniscus 4 mm below the defects until the meniscus far from them
         is 13.5 mm up (the defects end at 15), then drained back to the start
         (arclength in the volume, steps of 6, topology=True, trap=False):
         OUT/NAME.npz (each step's volume, pressure and contact line, and its
         mesh) and OUT/NAME.log. An hour or two each.
  draw   from the runs single, pair13 and pair11 in OUT (or hele_shaw.npz, the
         steps kept from them): hele_shaw.png (the pressure and where the
         contact line is against the volume, and the jumps under pressure
         control) and hele_shaw_shapes.png; with --film NAME also
         hele_shaw.mp4.

Everything but OUT goes into docs/_static.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

from pysurfaceevolver import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
sys.path.insert(0, HERE)
import hele_shaw_helpers as H  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
RUNS = (("single", (0.0,), "one defect"), ("pair13", (6.5,), "two, 13 mm apart"),
        ("pair11", (5.5,), "two, 11 mm apart"))
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED = "#333333", "#888888"


def _axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


# ---- a run ----------------------------------------------------------------------

def run(name, centres, out, ds=6.0, hmax=13.5, levels=2, sharp=False, rounding=H.S,
        halvings=6):
    """Imbibition from a flat meniscus at y = -4 until the meniscus far from
    the defects is at y = hmax, then drainage back to the start; rows
    (imbibing, volume added, pressure, contact line far from the defects,
    along the first defect's middle, fell, changes, facets)."""
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, f"{name}.log"), "w")
    t0 = time.time()

    def say(s):
        log.write(f"{time.time() - t0:8.1f}s {s}\n")
        log.flush()

    class Handler(logging.Handler):
        def emit(self, record):
            say("  . " + record.getMessage())

    logging.getLogger("pysurfaceevolver.topology").addHandler(Handler())
    logging.getLogger("pysurfaceevolver.topology").setLevel(logging.DEBUG)
    ev = H.cell(centres, sharp=sharp, rounding=rounding)
    say(f"start: {ev.relax(levels=levels)}")
    V0 = ev.body(1).volume
    rows, frames = [], {}

    def save():
        np.savez(os.path.join(out, f"{name}.npz"), rows=np.array(rows, float),
                 centres=np.array(centres),
                 rounding=np.array(rounding),
                 **{f"x{k}": v[0] for k, v in frames.items()},
                 **{f"f{k}": v[1] for k, v in frames.items()})

    for leg, sign in (("imbibition", 1), ("drainage", -1)):
        for attempt in range(4):
            finished = False
            for s in recipes.arclength(ev, ev.body(1), sign*ds, 100000, topology=True,
                                       trap=False, halvings=halvings):
                rows.append((sign > 0, s.value - V0, s.pressures[0],
                             H.contact_line(ev, H.X), H.contact_line(ev, centres[0]),
                             bool(s.fell), len(s.changes or ()), ev.counts["facets"]))
                say(f"{leg} V {s.value - V0:9.3f} p {s.pressures[0]:.5f} far {rows[-1][3]:.3f} "
                    f"mid {rows[-1][4]:.3f} facets {ev.counts['facets']}"
                    + (f" fell ({', '.join(c.kind for c in s.changes)})" if s.fell else ""))
                m = ev.mesh()
                frames[len(rows) - 1] = (np.asarray(m.vertices, np.float32), np.asarray(m.facets))
                if len(rows) % 10 == 0:
                    save()
                finished = (rows[-1][3] > hmax) if sign > 0 else (s.value - V0 < 0)
                if finished:
                    break
            if finished:
                break
            # (the branch ended short: relax where it is and go on)
            say(f"{leg} ended short; relaxing and going on ({attempt + 1})")
            say(f"  {ev.relax()}")
    save()
    say("done")

# ---- the pictures -----------------------------------------------------------------

def _data(out):
    """{name: rows} from the runs in out, kept in hele_shaw.npz (or from it)."""
    path = os.path.join(STATIC, "hele_shaw.npz")
    found = {name: os.path.join(out, f"{name}.npz") for name, _, _ in RUNS}
    if not all(os.path.exists(f) for f in found.values()):
        z = np.load(path)
        return {name: z[name] for name, _, _ in RUNS}
    data = {name: np.load(f)["rows"] for name, f in found.items()}
    np.savez(path, **data)
    return data


def _kept(V):
    """The steps a run kept: not those a later back-up went back over."""
    keep = np.ones(len(V), bool)
    for i in range(len(V) - 1, 0, -1):
        if keep[i]:
            keep[:i] &= ~(V[:i] > V[i] + 1e-9)
    return keep


def trend(rows):
    """The pressure of the meniscus undisturbed, a line in the volume fitted
    while the defects are far (the first 100 mm^3)."""
    r = rows[(rows[:, 0] > 0) & (rows[:, 1] < 100)]
    return np.polyfit(r[:, 1], r[:, 2], 1)


def controlled(rows):
    """Each leg under pressure control: (V, P, far, mid, on) for imbibition and
    drainage, ``on`` false at the states a jump passes over, and the jumps."""
    legs = []
    for up in (True, False):
        r = rows[(rows[:, 0] > 0) == up]
        r = r[_kept(r[:, 1] if up else -r[:, 1])]
        V, P = r[:, 1], r[:, 2]
        env, jumps = H.envelope(V, P, up)
        on = np.ones(len(V), bool)
        for a, b, *_ in jumps:
            on[a + 1:b] = False
        legs.append((V, P, r[:, 3], r[:, 4], on, jumps))
    return legs


LIQUID = dict(color="#3d8fd1", smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.2)
PLATE = dict(color="#b0b0b0", opacity=0.22, ambient=0.4, diffuse=0.6)
CAMERA = [(-26.0, -40.0, 34.0), (0.0, 6.0, 1.0), (0, 0, 1)]


def scene(pl, x, f, centres, label="", rounding=H.S):
    """The meniscus and its mirror image across x = 0, under the upper plate."""
    import pyvista as pv
    x = np.asarray(x, float)
    tri = np.hstack([np.full((len(f), 1), 3), f]).ravel()
    for s in (1, -1):
        m = pv.PolyData(x*np.array([s, 1, 1]), tri)
        pl.add_mesh(m, **LIQUID)
    xs, ys = np.linspace(-H.X, H.X, 161), np.linspace(-6, 22, 113)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = np.asarray(__import__("pysurfaceevolver").constraints._height_values(
        H.height(centres, rounding), gx.ravel(), gy.ravel()))
    pl.add_mesh(pv.StructuredGrid(gx, gy, np.asarray(gz).reshape(gx.shape)), **PLATE)
    if label:
        pl.add_text(label, font_size=11)


def frame(out, name, i):
    z = np.load(os.path.join(out, f"{name}.npz"))
    return z[f"x{i}"], z[f"f{i}"], tuple(z["centres"]), _rounding(z)


def _rounding(z):
    """The run's plate rounding (runs before --rounding: the default)."""
    return float(z["rounding"]) if "rounding" in z.files else H.S


def film(out, name, path=None, frames=240):
    """The run's kept steps (at most ``frames`` of them, evenly) as an mp4."""
    import drainage_topology_figure as D
    z = np.load(os.path.join(out, f"{name}.npz"))
    rows, centres = z["rows"], tuple(z["centres"])
    order = []
    for up in (True, False):
        idx = np.flatnonzero((rows[:, 0] > 0) == up)
        idx = idx[_kept(rows[idx, 1] if up else -rows[idx, 1])]
        order += [i for i in idx if f"x{i}" in z.files]
    if len(order) > frames:
        # (evenly in the oil moved, so the meniscus moves at a steady pace: the
        # runs' own steps bunch up near events)
        moved = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(rows[order, 1])))])
        pick = sorted({int(np.argmin(np.abs(moved - m)))
                       for m in np.linspace(0, moved[-1], frames)})
        order = [order[j] for j in pick]
    title = dict((n, t) for n, _, t in RUNS).get(name, name)

    def draw(pl, i):
        leg = "imbibition" if rows[i, 0] > 0 else "drainage"
        scene(pl, z[f"x{i}"], z[f"f{i}"], centres,
              f"{title}: {leg}, oil added {rows[i, 1]:.0f} mm^3, "
              f"pressure {20*rows[i, 2]:.1f} Pa", _rounding(z))

    D.movie(path or os.path.join(STATIC, f"hele_shaw_{name}.mp4"), order, draw, CAMERA,
            fps=24, hold=24, window=(960, 640), quality=5)


def draw(out, film=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = _data(out)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.6), constrained_layout=True,
                            sharey="row")
    for col, (name, centres, title) in enumerate(RUNS):
        rows = data[name]
        line = trend(rows)
        for (V, P, far, mid, on, jumps), colour, leg in zip(
                controlled(rows), COLOURS[:2], ("imbibition", "drainage")):
            eta = mid - far
            # (the pressure-controlled path: the traced states it passes
            # through, joined; a jump dashed from where it leaves to where it lands)
            axs[0, col].plot(np.where(on, far, np.nan), np.where(on, eta, np.nan),
                             color=colour, linewidth=1.8, label=leg)
            dp = P - np.polyval(line, V)
            axs[1, col].plot(V, dp, color=colour, linewidth=0.8, alpha=0.5)
            axs[1, col].plot(np.where(on, V, np.nan), np.where(on, dp, np.nan), color=colour,
                             linewidth=1.8, label=leg)
            for a, b, p, va, vb, e in jumps:
                axs[0, col].plot(far[[a, b]], eta[[a, b]], "--", color=colour, linewidth=1)
                axs[1, col].plot([va, vb], [p - np.polyval(line, va), p - np.polyval(line, vb)],
                                 "--", color=colour, linewidth=1)
        axs[0, col].set_title(title, color=INK)
        axs[0, col].set_xlabel("far contact line $h_f$ (mm)", color=INK)
        axs[1, col].set_xlabel("oil added (mm$^3$)", color=INK)
        for ax in axs[:, col]:
            _axes(ax)
    axs[0, 0].set_ylabel("bulge over the defect $\\eta_m$ (mm)", color=INK)
    axs[1, 0].set_ylabel("pressure over the undisturbed\nmeniscus's ($\\gamma$/mm = 20 Pa)",
                         color=INK)
    axs[0, 0].legend(frameon=False)
    fig.savefig(os.path.join(STATIC, "hele_shaw.png"), dpi=150)
    plt.close(fig)
    print("wrote hele_shaw.npz and hele_shaw.png")
    for name, centres, _ in RUNS:
        for up, leg in ((True, "imbibition"), (False, "drainage")):
            for a, b, p, va, vb, e in controlled(data[name])[0 if up else 1][5]:
                print(f"{name} {leg}: jump at p {p:.4f}, V {va:.1f} -> {vb:.1f}, "
                      f"dissipated {e:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=("run", "draw", "film"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="hele_shaw_runs")
    ap.add_argument("--film")
    ap.add_argument("--rounding", type=float, default=H.S,
                    help="the defects' edges rounded over this (mm; run)")
    ap.add_argument("--sharp", action="store_true",
                    help="the defects as blocks with sharp edges (run)")
    ap.add_argument("--levels", type=int, default=2,
                    help="refinement levels of the start (run)")
    ap.add_argument("--ds", type=float, default=6.0, help="the arclength step (run)")
    ap.add_argument("--halvings", type=int, default=6,
                    help="the arclength step's halvings at the most (run)")
    a = ap.parse_args()
    if a.part == "run":
        run(a.args[0], tuple(float(c) for c in a.args[1:]), a.out, sharp=a.sharp,
            rounding=a.rounding, levels=a.levels, ds=a.ds, halvings=a.halvings)
    elif a.part == "film":
        for name in a.args:
            film(a.out, name)
    else:
        draw(a.out, a.film)

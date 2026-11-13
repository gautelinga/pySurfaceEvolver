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
         control), hele_shaw_compare.png (the runs measured where the meniscus
         crosses half the gap, against the experiment in
         hele_shaw_experiment.csv and the line model, which takes a few
         minutes the first time) and hele_shaw_shapes.png; with --film NAME
         also hele_shaw.mp4.

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

def _save_state(ev, path, saved):
    """The surface (path.fe) and what pySE and the run remember of it
    (path.pkl), to go on from with run(resume=path)."""
    import pickle
    from pysurfaceevolver._engine import surface
    ev.dump(path + ".fe")
    with open(path + ".pkl", "wb") as f:
        pickle.dump(dict(saved, walls=dict(surface.walls), retired=set(surface.retired),
                         energy_shift=list(surface.energy_shift),
                         edge_scale=surface.edge_scale, sizes=dict(surface.sizes)), f)


def _restore_state(path):
    """A state _save_state wrote: the surface loaded, pySE's memory of it put
    back; and the run's own part."""
    import pickle
    import pysurfaceevolver as pyse
    from pysurfaceevolver._engine import surface
    ev = pyse.Evolver()
    ev.load(path + ".fe")
    with open(path + ".pkl", "rb") as f:
        saved = pickle.load(f)
    surface.walls.clear()
    surface.walls.update(saved.pop("walls"))
    surface.retired |= saved.pop("retired")
    surface.energy_shift[:] = saved.pop("energy_shift")
    surface.edge_scale = saved.pop("edge_scale")
    surface.sizes = dict(saved.pop("sizes"))
    ev._register_native()
    return ev, saved


def _trace_solid(box, say):
    """Log each solid event found with its middle in box (x0, x1, y0, y1):
    its gap, the edge length h there, the distance s from its closest vertex
    to the nearest vertex on that wall (a contact line), the gap over the
    straight wedge's s sin(angle), and the wall's curvatures there times h
    (a heightfield: the larger principal one, and the signed mean)."""
    from pysurfaceevolver import _events as E
    from pysurfaceevolver import constraints as C
    from pysurfaceevolver._engine import surface
    inner = E._solid
    x0, x1, y0, y1 = box

    def traced(ev, m, x, e, h, normal, behind, ahead, with_bodies, moving, factor,
               nums, attrs, dist, names):
        found = inner(ev, m, x, e, h, normal, behind, ahead, with_bodies, moving, factor,
                      nums, attrs, dist, names)
        for event in found:
            w = event.where
            if not (x0 <= w[0] <= x1 and y0 <= w[1] <= y1):
                continue
            c = [i for i in range(len(nums)) if (names[i] or int(nums[i])) == event.constraint]
            if not c:
                continue
            col = dist[:, c[0]]
            members = np.asarray(event.vertices)
            k = members[np.argmin(np.where(np.isnan(col[members]), np.inf, np.abs(col[members])))]
            on = np.flatnonzero(np.isnan(col))
            s = float(np.min(np.linalg.norm(x[on] - x[k], axis=1))) if len(on) else np.inf
            wall = surface.walls.get(event.constraint)
            angle = E._contact_angle(ev, wall)
            wedge = s*np.sin(np.radians(angle if event.between != "outside" else 180 - angle)) \
                if angle is not None else np.nan
            shape = getattr(wall, "shape", None)
            kap = bend = np.nan
            if shape is not None and shape[0] == "heightfield":
                d = 1e-2*float(h[k])
                kap = float(C._height_curvature(shape[1], x[k:k + 1, 0], x[k:k + 1, 1], d)[0])
                bend = float(C._height_bend(shape[1], x[k:k + 1, 0], x[k:k + 1, 1], d)[0])
            say(f"  solid-trace gap {event.gap:.3g} h {h[k]:.3g} s/h {s/h[k]:.2f} "
                f"gap/wedge {event.gap/wedge if wedge else np.nan:.3g} kappa*h {kap*h[k]:.3g} "
                f"bend*h {bend*h[k]:.3g} n {len(members)} {event.between} "
                f"at ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})")
        return found
    E._solid = traced


def run(name, centres, out, ds=6.0, hmax=13.5, levels=2, sharp=False, rounding=H.S,
        halvings=6, width=H.X, states=0, resume=None, trace=None, dumps=0, fine=1.0,
        crossed=0):
    """Imbibition from a flat meniscus at y = -4 until the meniscus far from
    the defects is at y = hmax, then drainage back to the start; rows
    (imbibing, volume added, pressure, contact line far from the defects,
    along the first defect's middle, fell, changes, facets). ``states``: keep
    the last so many states (every fifth step has one) in
    OUT/states/NAME/stepNNNN.{fe,pkl}; ``resume``: such a state (without the
    extension), the run going on from it exactly (its rows up to there kept;
    ds, halvings and width as it had them). ``trace``: (x0, x1, y0, y1), the
    solid events there logged (_trace_solid), and the surface just after
    each of the first ``dumps`` changes made there dumped to
    OUT/changes/NAME-NNN.fe. ``fine``: the start's vertices that many times as
    close, across the cell and the gap (the run's edges follow the start's).
    ``crossed``: the surface of the first so many runs judged to have passed
    it through itself dumped as it was then, OUT/crossed/NAME-NNN.fe, and as
    the last remeshing or coarsening before left it, NAME-NNN-before.fe."""
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, f"{name}.log"), "a" if resume else "w")
    t0 = time.time()

    def say(s):
        log.write(f"{time.time() - t0:8.1f}s {s}\n")
        log.flush()

    live, dumped, crossings = [], [0], [0]

    class Handler(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            say("  . " + text)
            folder = os.path.join(out, "crossed")
            if live and crossings[0] < crossed and text.startswith(("remeshed", "coarsened")):
                # (the surface the next run starts from, kept till a run is judged crossed)
                os.makedirs(folder, exist_ok=True)
                live[0].dump(os.path.join(folder, f"{name}-last.fe"))
            if live and crossings[0] < crossed and text.startswith("implausible run: crossings"):
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, f"{name}-{crossings[0]:03d}.fe")
                live[0].dump(path)
                for tag in ("", "-guarded", "-tidied", "-after-run", "-after-newton",
                            "-after-split", "-after-refine"):
                    last = os.path.join(folder, f"{name}-last{tag}.fe")
                    if os.path.exists(last):
                        os.replace(last, os.path.join(
                            folder, f"{name}-{crossings[0]:03d}-before{tag}.fe"))
                say(f"  dumped {path}")
                crossings[0] += 1
            if trace and dumps and live and dumped[0] < dumps and text.startswith("change made"):
                import re
                at = re.search(r" at \(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)", text)
                if at and trace[0] <= float(at[1]) <= trace[1] \
                        and trace[2] <= float(at[2]) <= trace[3]:
                    folder = os.path.join(out, "changes")
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.join(folder, f"{name}-{dumped[0]:03d}.fe")
                    live[0].dump(path)
                    say(f"  dumped {path}")
                    dumped[0] += 1

    if crossed:
        # (and the surface after each guard() and _tidy(), the latest kept,
        # beside the next crossed run's dumps: which call left it as it was)
        from pysurfaceevolver import _surgery

        def keeping(f, tag):
            def kept(ev, *args, **kw):
                out_ = f(ev, *args, **kw)
                if live and crossings[0] < crossed:
                    folder = os.path.join(out, "crossed")
                    os.makedirs(folder, exist_ok=True)
                    live[0].dump(os.path.join(folder, f"{name}-last-{tag}.fe"))
                return out_
            return kept
        _surgery.guard = keeping(_surgery.guard, "guarded")
        _surgery._tidy = keeping(_surgery._tidy, "tidied")
        # (inside a descent's runs: which of relax()'s stages adds vertices, each
        # logged with the counts and the smallest facet angle after it, the
        # surface after each kept)
        import pysurfaceevolver as _pyse

        def smallest(ev):
            m = ev.mesh()
            x, f = np.asarray(m.vertices), np.asarray(m.facets)
            a, b, c = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
            out_ = 180.0
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                p1, p2 = v - u, w - u
                cos = (p1*p2).sum(1)/np.maximum(
                    np.linalg.norm(p1, axis=1)*np.linalg.norm(p2, axis=1), 1e-300)
                out_ = min(out_, float(np.degrees(np.arccos(np.clip(cos, -1, 1))).min()))
            return out_

        inside = [0]

        def watched(f, tag):
            def run_(self_or_ev, *args, **kw):
                ev = self_or_ev
                if not (live and crossings[0] < crossed) or (tag != "run" and not inside[0]):
                    return f(self_or_ev, *args, **kw)
                n0 = ev.counts["vertices"]
                inside[0] += tag == "run"
                try:
                    out_ = f(self_or_ev, *args, **kw)
                finally:
                    inside[0] -= tag == "run"
                say(f"  [{tag}] vertices {n0} -> {ev.counts['vertices']}, smallest angle "
                    f"{smallest(ev):.2f}" + (f", {out_}" if tag in ("run", "newton") else ""))
                folder = os.path.join(out, "crossed")
                os.makedirs(folder, exist_ok=True)
                ev.dump(os.path.join(folder, f"{name}-last-after-{tag}.fe"))
                return out_
            return run_
        _surgery._run = watched(_surgery._run, "run")
        _pyse.Evolver.newton = watched(_pyse.Evolver.newton, "newton")
        _pyse.Evolver._split_marked = watched(_pyse.Evolver._split_marked, "split")
        _pyse.Evolver.refine = watched(_pyse.Evolver.refine, "refine")
    logging.getLogger("pysurfaceevolver.topology").addHandler(Handler())
    logging.getLogger("pysurfaceevolver.topology").setLevel(logging.DEBUG)
    if resume:
        ev, saved = _restore_state(resume)
        V0, legs, carry = saved["V0"], saved["legs"], saved["run"]
        ds, halvings, width = saved["ds"], saved["halvings"], saved["width"]
        rows = saved["rows"]
        say(f"resumed from {resume}: step {len(rows) - 1}, {legs[0][0]}")
    else:
        # (the start's vertices as far apart across as at the full width)
        ev = H.cell(centres, sharp=sharp, rounding=rounding, width=width,
                    nx=int(round(40*fine*width/H.X)) + 1, nz=int(round(6*fine)) + 1)
        say(f"start: {ev.relax(levels=levels)}")
        V0 = ev.body(1).volume
        rows, carry = [], None
        legs = (("imbibition", 1), ("drainage", -1))
    live.append(ev)
    if trace:
        _trace_solid(trace, say)
    frames = {}
    kept, held = os.path.join(out, "states", name), []
    if states:
        os.makedirs(kept, exist_ok=True)

    def save():
        np.savez(os.path.join(out, f"{name}.npz"), rows=np.array(rows, float),
                 centres=np.array(centres),
                 rounding=np.array(rounding), width=np.array(width),
                 **{f"x{k}": v[0] for k, v in frames.items()},
                 **{f"f{k}": v[1] for k, v in frames.items()})

    for k, (leg, sign) in enumerate(legs):
        for attempt in range(4):
            finished = False
            for s in recipes.arclength(ev, ev.body(1), sign*ds, 100000, topology=True,
                                       trap=False, halvings=halvings, resumable=bool(states),
                                       resume=carry):
                rows.append((sign > 0, s.value - V0, s.pressures[0],
                             H.contact_line(ev, width), H.contact_line(ev, centres[0]),
                             bool(s.fell), len(s.changes or ()), ev.counts["facets"]))
                say(f"{leg} V {s.value - V0:9.3f} p {s.pressures[0]:.5f} far {rows[-1][3]:.3f} "
                    f"mid {rows[-1][4]:.3f} facets {ev.counts['facets']}"
                    + (f" fell ({', '.join(c.kind for c in s.changes)})" if s.fell else ""))
                m = ev.mesh()
                frames[len(rows) - 1] = (np.asarray(m.vertices, np.float32), np.asarray(m.facets))
                if len(rows) % 10 == 0:
                    save()
                own = s.state() if states and s.state else None
                if own is not None:
                    # (every fifth step carries the run's state)
                    path = os.path.join(kept, f"step{len(rows) - 1:04d}")
                    _save_state(ev, path, dict(V0=V0, rows=list(rows), legs=legs[k:], run=own,
                                                    ds=ds, halvings=halvings, width=width))
                    held.append(path)
                    while len(held) > states:
                        stale = held.pop(0)
                        for ext in (".fe", ".pkl"):
                            os.remove(stale + ext)
                finished = (rows[-1][3] > hmax) if sign > 0 else (s.value - V0 < 0)
                if finished:
                    break
            carry = None
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
    _keep(**data)
    return data


def _keep(**arrays):
    """Put arrays into hele_shaw.npz beside what it holds."""
    path = os.path.join(STATIC, "hele_shaw.npz")
    held = dict(np.load(path)) if os.path.exists(path) else {}
    held.update(arrays)
    np.savez_compressed(path, **held)


def _compared(out):
    """For each cell: its run seen through the plates, rows (imbibition, V, h_f,
    eta_m) where the meniscus crosses half the gap; the line model in the run's
    cell at the slope here, and in a wide cell at the design's 6 kPa/m. Kept in
    hele_shaw.npz."""
    path = os.path.join(STATIC, "hele_shaw.npz")
    held = dict(np.load(path)) if os.path.exists(path) else {}
    new = {}
    for name, centres, _ in RUNS:
        run_file = os.path.join(out, f"{name}.npz")
        if f"{name}_seen" not in held or os.path.exists(run_file):
            z = np.load(run_file)
            rows, width, rounding = z["rows"], float(z["width"]), float(z["rounding"])
            seen = []
            for k in range(len(rows)):
                v = z[f"x{k}"]
                far = H.half_gap(v, width - 0.15, centres, rounding, band=0.15)
                seen.append((rows[k, 0], rows[k, 1], far,
                             H.half_gap(v, centres[0], centres, rounding) - far))
            new[f"{name}_seen"] = np.array(seen)
            if f"{name}_cell" not in held or float(held.get(f"{name}_width", 0)) != width:
                new[f"{name}_cell"] = H.line_model(centres, width, rounding=rounding)
                new[f"{name}_width"] = np.array(width)
        if f"{name}_wide" not in held:
            new[f"{name}_wide"] = H.line_model(centres, 40.0, slope=H.PI)
    if new:
        new = {k: np.asarray(v, np.float32) for k, v in new.items()}
        _keep(**new)
        held.update(new)
    return held


def _experiment():
    """{(cell, leg): (h_f, eta_m)} from hele_shaw_experiment.csv."""
    got = {}
    with open(os.path.join(STATIC, "hele_shaw_experiment.csv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("cell,"):
                continue
            cell, leg, hf, eta = line.strip().split(",")
            got.setdefault((cell, leg), []).append((float(hf), float(eta)))
    return {k: np.array(v) for k, v in got.items()}


def draw_compare(out):
    """hele_shaw_compare.png: each cell's bulge against the far meniscus, all
    measured where the meniscus crosses half the gap: the run, the experiment
    and the line model."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    held, seen_exp = _compared(out), _experiment()
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.1), sharey=True)
    for ax, (name, centres, title) in zip(axs, RUNS):
        for up, leg, colour in ((True, "imbibition", COLOURS[0]), (False, "drainage", COLOURS[1])):
            r = held[f"{name}_seen"]
            r = r[(r[:, 0] > 0) == up]
            r = r[_kept(r[:, 1] if up else -r[:, 1])]
            ax.plot(r[:, 2], r[:, 3], color=colour, linewidth=1.6, zorder=3)
            for key, style, width in (("wide", "--", 1.0), ("cell", ":", 1.3)):
                m = held[f"{name}_{key}"]
                m = m[(m[:, 0] > 0) == up]
                ax.plot(m[:, 1], m[:, 2], style, color=colour, linewidth=width, zorder=2)
            e = seen_exp[(name, leg)]
            ax.plot(e[:, 0], e[:, 1], "o", color=colour, markersize=2.2, alpha=0.6,
                    markeredgewidth=0, zorder=4)
        ax.set_title(title, color=INK)
        ax.set_xlabel("far meniscus $h_f$ (mm)", color=INK)
        ax.set_xlim(-3.5, 16)
        ax.set_ylim(-0.5, 10.8)
        _axes(ax)
    axs[0].set_ylabel("bulge over the defect $\\eta_m$ (mm)", color=INK)
    grey = MUTED
    handles = [Line2D([], [], color=COLOURS[0], linewidth=2, label="imbibition"),
               Line2D([], [], color=COLOURS[1], linewidth=2, label="drainage"),
               Line2D([], [], color=grey, linewidth=1.6, label="pySE"),
               Line2D([], [], color=grey, linestyle="none", marker="o", markersize=4,
                      label="experiment"),
               Line2D([], [], color=grey, linestyle=":", linewidth=1.3,
                      label="line model, these cells"),
               Line2D([], [], color=grey, linestyle="--", linewidth=1.0,
                      label="line model, wide cell, 6 kPa/m")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(STATIC, "hele_shaw_compare.png"), dpi=150)
    plt.close(fig)
    print("wrote hele_shaw_compare.png")


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


def scene(pl, x, f, centres, label="", rounding=H.S, width=H.X):
    """The meniscus and its mirror image across x = 0, under the upper plate
    (out to the run's mirror planes, x = -width and width)."""
    import pyvista as pv
    x = np.asarray(x, float)
    tri = np.hstack([np.full((len(f), 1), 3), f]).ravel()
    for s in (1, -1):
        m = pv.PolyData(x*np.array([s, 1, 1]), tri)
        pl.add_mesh(m, **LIQUID)
    xs, ys = np.linspace(-width, width, int(round(8*width)) + 1), np.linspace(-6, 22, 113)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = np.asarray(__import__("pysurfaceevolver").constraints._height_values(
        H.height(centres, rounding), gx.ravel(), gy.ravel()))
    pl.add_mesh(pv.StructuredGrid(gx, gy, np.asarray(gz).reshape(gx.shape)), **PLATE)
    if label:
        pl.add_text(label, font_size=11)


def frame(out, name, i):
    z = np.load(os.path.join(out, f"{name}.npz"))
    return z[f"x{i}"], z[f"f{i}"], tuple(z["centres"]), _rounding(z), _width(z)


def _rounding(z):
    """The run's plate rounding (runs before --rounding: the default)."""
    return float(z["rounding"]) if "rounding" in z.files else H.S


def _width(z):
    """The run's half-width (runs before --width: the default)."""
    return float(z["width"]) if "width" in z.files else H.X


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
              f"pressure {20*rows[i, 2]:.1f} Pa", _rounding(z), _width(z))

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
    draw_compare(out)
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
    ap.add_argument("--width", type=float, default=H.X,
                    help="the cell's half-width modelled, mirror planes x = 0 and x = width (run)")
    ap.add_argument("--crossed", type=int, default=0,
                    help="dump the surface of the first so many runs judged crossed, "
                         "OUT/crossed (run)")
    ap.add_argument("--fine", type=float, default=1.0,
                    help="the start's vertices this many times as close (run)")
    ap.add_argument("--halvings", type=int, default=6,
                    help="the arclength step's halvings at the most (run)")
    ap.add_argument("--states", type=int, default=0,
                    help="keep the last so many states (every fifth step's), OUT/states/NAME (run)")
    ap.add_argument("--resume",
                    help="go on from such a state, its path without .fe/.pkl (run)")
    ap.add_argument("--trace", help="x0,x1,y0,y1: the solid events there logged (run)")
    ap.add_argument("--dumps", type=int, default=0,
                    help="with --trace: the surface after the first so many changes "
                         "there dumped, OUT/changes (run)")
    a = ap.parse_args()
    if a.part == "run":
        run(a.args[0], tuple(float(c) for c in a.args[1:]), a.out, sharp=a.sharp,
            rounding=a.rounding, levels=a.levels, ds=a.ds, halvings=a.halvings,
            width=a.width, states=a.states, resume=a.resume,
            trace=tuple(float(v) for v in a.trace.split(",")) if a.trace else None,
            dumps=a.dumps, fine=a.fine, crossed=a.crossed)
    elif a.part == "film":
        for name in a.args:
            film(a.out, name)
    else:
        draw(a.out, a.film)

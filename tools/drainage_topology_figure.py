"""Draw the pictures and movies of docs/drainage_topology.md. Needs matplotlib,
PyVista (off-screen rendering) and imageio-ffmpeg (the movies).

    OMP_NUM_THREADS=1 python tools/drainage_topology_figure.py [part ...]

Parts (all by default):

  drop   the drop draining over a sphere: drainage_drop.mp4 (a minute)
  pore   one pore of a bead pack (four unit beads, contact angle 20):
         drainage_pore.png (the liquid's curvature against its volume, the
         changes of topology marked), drainage_pore_shapes.png (just before the
         jump, after it and after the fall that follows) and drainage_pore.mp4
         (a minute)
  ring   two beads stacked, drained through the floor, the ring between them
         cut off: drainage_ring.png, drainage_ring_shapes.png and
         drainage_ring.mp4 (half a minute)
  pack   a 16-bead pack to breakthrough, from a series that
         bench/topology/series.py wrote (--series DIR; made with
         ``series.py --seed 2 --theta 20 --frames 150``, about ten minutes):
         drainage_pack.mp4 and drainage_pack_shapes.png
  pack64 a run of 64 beads in a periodic cell to breakthrough, from the
         snapshots bench/topology/snapshots.py kept of it (--snaps DIR), the
         cascades too: drainage_pack64.mp4 and drainage_pack64_shapes.png
  pc64   the capillary pressure against the saturation for that set, from
         corpus.py's output folder (--runs OUT) into drainage_pc64.npz, or from
         that file: drainage_pc64.png
  periodic  the two beads in a periodic cell (periodic.md): periodic_ring.mp4,
         periodic_ring_shapes.png (the cell tiled 2 x 2)
  packs  the wide set's runs (bench/topology/corpus.py --set wide), from their
         logs (--logs DIR) into drainage_packs.npz, or from that file:
         drainage_packs.png (the trapped liquid as each run drains)

Everything goes into docs/_static.
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
from packing_helpers import _pore_surface, _on  # noqa: E402
import ring_helpers  # noqa: E402
import periodic_helpers  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
CENTRES = np.array([[-0.6123, 1.1001, 0.3757], [1.2879, 0.1557, 0.1317],
                    [-0.4315, -1.0784, 0.651], [-0.2441, -0.1774, -1.1584]])
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.2,
              smooth_shading=False, ambient=0.25, diffuse=0.75)
TRAPPED = dict(LIQUID, color="#fdbf6f", edge_color="#e09a3e")
BEAD = dict(color="sandybrown", opacity=0.25)
FAINT = dict(color="#c8c0b4", opacity=0.1)     # (the 64 beads, the liquid seen through them)
WINDOW = (960, 720)


def polydata(x, f):
    import pyvista as pv
    return pv.PolyData(np.asarray(x, float), np.c_[np.full(len(f), 3), f])


def liquid(p, x, f, body=None):
    """The liquid on the plotter: the draining body blue, the trapped ones
    (body > 1) orange."""
    f = np.asarray(f)
    if body is None:
        p.add_mesh(polydata(x, f), **LIQUID)
        return
    body = np.asarray(body)
    for keep, style in ((body <= 1, LIQUID), (body > 1, TRAPPED)):
        if keep.any():
            p.add_mesh(polydata(x, f[keep]), **style)


def movie(path, frames, draw, camera, fps=12, hold=24, window=WINDOW, quality=7):
    """An mp4 of draw(plotter, frame) over the frames, the last held for
    ``hold`` frames (``quality`` 0-10, imageio's)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=window)
    p.open_movie(path, framerate=fps, quality=quality)
    for k, frame in enumerate(frames):
        p.clear()
        draw(p, frame)
        p.camera_position = camera
        p.write_frame()
        if k == len(frames) - 1:
            for _ in range(hold):
                p.write_frame()
    p.close()
    print(f"wrote {os.path.relpath(path)} ({len(frames)} frames)")


def run_steps(ev, value, ds, n, stop, **options):
    """arclength(topology=True)'s steps and the mesh (and each facet's
    body) at each."""
    steps, meshes = [], []
    for s in recipes.arclength(ev, value, ds, n, topology=True, **options):
        m = ev.mesh()
        steps.append(s)
        meshes.append((np.array(m.vertices), np.array(m.facets),
                       np.asarray(m.facet_bodies).max(axis=1)))
        if stop(ev, s):
            break
    return steps, meshes


# ---- the drop over a sphere -------------------------------------------------

def drop():
    warnings.simplefilter("ignore")
    cap = shapes.cap(3.0, 3.0, nr=8, nt=32)
    centre, radius = (1.3, 0, 1.0), 0.8
    ball = C.sphere(centre, radius, contact_angle=30.0, axis=(-1, 0, -1), wet_poles=("north",))
    ev = pyse.Evolver()
    ev.load_arrays(cap.vertices, cap.faces,
                   constraints={"plate": C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                "ball": ball},
                   vertex_constraints={"plate": cap.rims["rim"]},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.refine(1)
    ev.relax()
    steps, meshes = run_steps(ev, ev.body(1), -1.0, 60, lambda ev, s: s.value < 10)
    import pyvista as pv
    sphere = pv.Sphere(radius=radius, center=centre, theta_resolution=48, phi_resolution=48)
    plate = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=8, j_size=8)

    def draw(p, i):
        x, f, _ = meshes[i]
        liquid(p, x, f)
        p.add_mesh(sphere, color="sandybrown")
        p.add_mesh(plate, color="#dddddd")
        p.add_text(f"volume {steps[i].value:.1f}" + ("   (after a change)" if steps[i].fell
                                                     else ""), font_size=12)
    movie(os.path.join(STATIC, "drainage_drop.mp4"), range(len(steps)), draw,
          [(9.0, -7.0, 6.0), (0.6, 0, 0.9), (0, 0, 1)])


# ---- one pore ---------------------------------------------------------------

def pore_run():
    warnings.simplefilter("ignore")
    v, f = _pore_surface(CENTRES, 0.2, 0.1)
    beads = {f"bead{k + 1}": C.sphere(c, 1.0, contact_angle=20.0, axis=c)
             for k, c in enumerate(CENTRES)}
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=beads,
                   vertex_constraints={k: _on(v, c) for k, c in zip(beads, CENTRES)},
                   bodies=[pyse.Body(volume=1.0)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return run_steps(ev, ev.body(1), -0.01, 150, lambda ev, s: s.value < 0.2)


def pore_trace(steps):
    V = np.array([s.value for s in steps])
    k = np.array([-s.pressures[0] for s in steps])
    fell = np.array([s.fell for s in steps])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(V, k, "-", color="#1f77b4", lw=1.2, label="arclength points")
    ax.plot(V[fell], k[fell], "o", color="#d62728", ms=5, label="after a change of topology")
    ax.invert_xaxis()
    ax.set_xlabel("draining liquid's volume (drained to the right)")
    ax.set_ylabel("curvature of the liquid surface")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "drainage_pore.png"), dpi=150)
    plt.close(fig)


def pore_beads(p):
    import pyvista as pv
    for c in CENTRES:
        p.add_mesh(pv.Sphere(radius=1.0, center=c, theta_resolution=48, phi_resolution=48),
                   **BEAD)


def pore_shapes(steps, meshes):
    import pyvista as pv
    pv.OFF_SCREEN = True
    k = np.array([-s.pressures[0] for s in steps])
    jump = max((i for i, s in enumerate(steps) if s.fell), key=lambda i: k[i - 1] - k[i])
    again = next(i for i in range(jump + 1, len(steps)) if steps[i].fell)
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 700), border=False)
    for col, i in enumerate((jump - 1, jump, again)):
        p.subplot(0, col)
        x, f, body = meshes[i]
        liquid(p, x, f, body)
        pore_beads(p)
        p.add_text(f"volume {steps[i].value:.2f}, curvature {k[i]:.1f}"
                   + (f", cut off {steps[i].trapped:.2f}" if steps[i].trapped else ""),
                   font_size=12)
        p.camera_position = [(5.0, 4.0, 3.0), (0, 0, 0), (0, 0, 1)]
    p.screenshot(os.path.join(STATIC, "drainage_pore_shapes.png"))
    p.close()


def pore():
    steps, meshes = pore_run()
    pore_trace(steps)
    pore_shapes(steps, meshes)
    k = np.array([-s.pressures[0] for s in steps])

    def draw(p, i):
        x, f, body = meshes[i]
        liquid(p, x, f, body)
        pore_beads(p)
        p.add_text(f"volume {steps[i].value:.2f}, curvature {k[i]:.1f}"
                   + (f", cut off {steps[i].trapped:.2f}" if steps[i].trapped else "")
                   + ("   (after a change)" if steps[i].fell else ""), font_size=12)
    movie(os.path.join(STATIC, "drainage_pore.mp4"), range(len(steps)), draw,
          [(5.0, 4.0, 3.0), (0, 0, 0), (0, 0, 1)])
    print("wrote drainage_pore.png and drainage_pore_shapes.png")


# ---- a ring cut off ---------------------------------------------------------

def ring_beads(p):
    import pyvista as pv
    for c in ring_helpers.BEADS:
        p.add_mesh(pv.Sphere(radius=1.0, center=c, theta_resolution=48, phi_resolution=48),
                   **BEAD)
    L = ring_helpers.L
    p.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, 4.2)), style="wireframe", color="#555555")
    p.add_mesh(pv.Plane(center=(L/2, L/2, 0), direction=(0, 0, 1), i_size=L, j_size=L),
               color="#dddddd")


RING_CAMERA = [(9.5, -6.5, 6.0), (1.5, 1.5, 1.9), (0, 0, 1)]


def ring():
    warnings.simplefilter("ignore")
    ev = ring_helpers.two_beads(theta=20.0)
    steps, meshes = run_steps(ev, ev.body(1), -0.5, 200, ring_helpers.at_floor,
                              outlet="bottom")
    V = np.array([s.value for s in steps])
    T = np.array([s.trapped for s in steps])
    k = np.array([-s.pressures[0] for s in steps])
    kr = np.array([-s.pressures[1] if len(s.pressures) > 1 else np.nan for s in steps])
    cut = int(np.flatnonzero(T > 0)[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(V, V, "-", color="#1f77b4", lw=1.2, label="draining liquid")
    a.plot(V[cut:], T[cut:], "-", color="#ff7f00", lw=1.6, label="the ring, cut off")
    a.axvline(V[cut], color="gray", ls=":", lw=1)
    a.invert_xaxis()
    a.set_xlabel("draining liquid's volume")
    a.set_ylabel("volume")
    a.set_yscale("log")
    a.legend(frameon=False, loc="center left")
    b.plot(V, k, "-", color="#1f77b4", lw=1.2, label="draining liquid")
    b.plot(V, kr, "-", color="#ff7f00", lw=1.6, label="the ring")
    b.axvline(V[cut], color="gray", ls=":", lw=1)
    b.invert_xaxis()
    b.set_xlabel("draining liquid's volume")
    b.set_ylabel("curvature of the liquid surface")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "drainage_ring.png"), dpi=150)
    plt.close(fig)
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1500, 600), border=False)
    for col, i in enumerate((cut - 1, cut, len(steps) - 1)):
        p.subplot(0, col)
        x, f, body = meshes[i]
        liquid(p, x, f, body)
        ring_beads(p)
        p.add_text(f"volume {V[i]:.2f}" + (f", ring {T[i]:.3f}" if T[i] else ""), font_size=11)
        p.camera_position = RING_CAMERA
    p.screenshot(os.path.join(STATIC, "drainage_ring_shapes.png"))
    p.close()

    def draw(p, i):
        x, f, body = meshes[i]
        liquid(p, x, f, body)
        ring_beads(p)
        p.add_text(f"draining liquid {V[i]:.2f}" + (f", ring {T[i]:.4f} (held)" if T[i] else ""),
                   font_size=12)
    movie(os.path.join(STATIC, "drainage_ring.mp4"), range(len(steps)), draw, RING_CAMERA,
          fps=6)
    print("wrote drainage_ring.png and drainage_ring_shapes.png")


# ---- a 16-bead pack ---------------------------------------------------------

def pack(series):
    import h5py
    import pyvista as pv
    (h5path,) = glob.glob(os.path.join(series, "*.h5"))
    base = h5path[:-3]
    beads = pv.read(base + "_beads.vtp")
    box = pv.read(base + "_box.vtp")
    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds
    L, top = xmax - xmin, zmax
    camera = [(3.1*L, -2.3*L, 2.4*L), (L/2, L/2, top/2), (0, 0, 1)]
    times = [float(t) for t in re.findall(r'<Time Value="([^"]+)"', open(base + ".xdmf").read())]
    with h5py.File(h5path) as h5:
        names = sorted(h5)
        frames = [(np.array(h5[n]["xyz"]), np.array(h5[n]["tri"]), np.array(h5[n]["body"]))
                  for n in names]

    def scene(p, i):
        x, f, body = frames[i]
        liquid(p, x, f, body)
        p.add_mesh(beads, **BEAD)
        p.add_mesh(box, style="wireframe", color="#555555")
        trapped = len(set(body[body > 1].tolist()))
        p.add_text(f"drained {times[i]:.0%}" + (f", {trapped} trapped" if trapped else ""),
                   font_size=12)

    movie(os.path.join(STATIC, "drainage_pack.mp4"), range(len(frames)), scene, camera)
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1500, 600), border=False)
    for col, i in enumerate((0, len(frames)//2, len(frames) - 1)):
        p.subplot(0, col)
        scene(p, i)
        p.camera_position = camera
    p.screenshot(os.path.join(STATIC, "drainage_pack_shapes.png"))
    p.close()
    print("wrote drainage_pack_shapes.png")


# ---- the wide set -----------------------------------------------------------

LINE = re.compile(r"step\s+(\d+)\s+V ([-\d.]+) trapped ([-\d.]+)")


def packs(logs):
    path = os.path.join(STATIC, "drainage_packs.npz")
    if logs:
        data = {}
        for log in sorted(glob.glob(os.path.join(logs, "n16_*.log"))):
            name = os.path.basename(log).split("_h")[0][4:]           # s0_t15
            rows = [(float(m[2]), float(m[3])) for m in map(LINE.search, open(log)) if m]
            data[name] = np.array(rows)
        np.savez_compressed(path, **data)
    data = dict(np.load(path))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    colours = {"t15": "#1b9e77", "t20": "#d95f02", "t25": "#7570b3"}
    for name, rows in sorted(data.items()):
        V, T = rows[:, 0], rows[:, 1]
        start = V[0] + T[0]
        theta = name.split("_")[1]
        label = f"{theta[1:]} degrees" if name.startswith("s0_") else None
        a.plot(1 - (V + T)/start, T/start, "-", color=colours[theta], lw=1, label=label)
        b.plot(1 - V[-1]/start - T[-1]/start, T[-1]/start, "o", color=colours[theta])
    a.set_xlabel("share of the liquid drained")
    a.set_ylabel("share trapped")
    a.legend(frameon=False, title="contact angle")
    b.set_xlabel("share drained at breakthrough")
    b.set_ylabel("share trapped at breakthrough")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "drainage_packs.png"), dpi=150)
    plt.close(fig)
    print(f"wrote drainage_packs.png ({len(data)} runs)")


# ---- 64 beads in a periodic cell ----------------------------------------------

def _state(path):
    """A state corpus.py kept (--frames): the run's Evolver, the liquid it
    started with, its step."""
    sys.path.insert(0, os.path.join(HERE, "..", "bench", "topology"))
    import corpus
    return corpus.restore_state(path)


def _centres():
    from pysurfaceevolver._engine import surface
    return np.array([np.asarray(w.slabflux[0], float) for w in surface.walls.values()
                     if getattr(w, "slabflux", None)])


def cell_beads(centres, L, top):
    """The beads and their images in the periodic cell, cut at its sides: one
    mesh."""
    import pyvista as pv
    bounds, parts = (0, L, 0, L, 0, top), []
    for centre in centres:
        for dx in (-L, 0, L):
            for dy in (-L, 0, L):
                cc = centre + (dx, dy, 0)
                if -1 < cc[0] < L + 1 and -1 < cc[1] < L + 1:
                    s = pv.Sphere(radius=1.0, center=cc, theta_resolution=24, phi_resolution=24)
                    s = s.clip_box(bounds, invert=False)
                    if s.n_points:
                        parts.append(s)
    return pv.merge(parts)


def cell_scene(p, corners, body, beads, L, top):
    """One periodic cell: the liquid's facets moved into it (each by its first
    corner's image), the beads, the floor."""
    import pyvista as pv
    c = np.array(corners, float)
    c[:, :, :2] -= (np.floor(c[:, 0, :2]/L)*L)[:, None, :]
    x = c.reshape(-1, 3)
    liquid(p, x, np.arange(len(x)).reshape(-1, 3), body)
    p.add_mesh(beads, **FAINT)
    p.add_mesh(pv.Plane(center=(L/2, L/2, 0), direction=(0, 0, 1), i_size=L, j_size=L),
               color="#dddddd")
    p.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, top)), style="wireframe", color="#555555")


def pack64(snaps, frames=360):
    """The film and shapes of a run from bench/topology/snapshots.py's
    snapshots, at ``frames`` equally spaced volumes of the liquid (the
    draining body and the trapped ones): each the state the run held at the
    nearest of its steps' volumes (after the fall, where it fell there: each
    an equilibrium the run converged to; the last the breakthrough, where the
    gas reached the floor inside the last fall, not an equilibrium)."""
    import pyvista as pv
    index = json.load(open(os.path.join(snaps, "index.json")))
    beads = np.load(os.path.join(snaps, "beads.npz"))
    centres, L = beads["centres"], float(beads["L"])
    top = float(centres[:, 2].max()) + 1.0
    held = {}                                   # liquid volume -> the last snapshot there
    end = index[-1][0] if index[-1][1] == "end" else None
    for n, kind, step, V, T in index:
        if kind != "run" and V is not None:
            held[round(V + T, 9)] = n
    liquid_at = {n: v for v, n in held.items()}
    v0 = max(held)
    volumes = np.array(sorted(held, reverse=True))
    targets = np.linspace(volumes[0], volumes[-1], frames)
    chosen = [held[volumes[int(np.argmin(abs(volumes - t)))]] for t in targets]

    def shot(n):
        z = np.load(os.path.join(snaps, f"s{n:06d}.npz"))
        body = z["b"].astype(int)
        return z["c"].astype(float), body, 1 - liquid_at[n]/v0, len(set(body[body > 1].tolist()))

    cell = cell_beads(centres, L, top)
    focus = np.array([L/2, L/2, top/2])
    look = np.array([1.0, -0.75, 0.3])
    camera = [tuple(focus + 2.4*top*look/np.linalg.norm(look)), tuple(focus), (0, 0, 1)]

    def scene(pl, n):
        c, body, drained, trapped = shot(n)
        cell_scene(pl, c, body, cell, L, top)
        pl.add_text(f"drained {drained:.0%}" + (f", {trapped} trapped" if trapped else "")
                    + ("\nbreakthrough (in the fall)" if n == end else ""), font_size=12)

    movie(os.path.join(STATIC, "drainage_pack64.mp4"), chosen, scene, camera, fps=24,
          hold=48, window=(540, 720))
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1500, 800), border=False)
    last = 1 - liquid_at[chosen[-1]]/v0
    middle = min(chosen, key=lambda n: abs(1 - liquid_at[n]/v0 - last/2))
    for col, n in enumerate((chosen[0], middle, chosen[-1])):
        p.subplot(0, col)
        scene(p, n)
        p.camera_position = camera
    p.screenshot(os.path.join(STATIC, "drainage_pack64_shapes.png"))
    p.close()
    print("wrote drainage_pack64_shapes.png")


def pores(state):
    """A periodic pack's inlet plane z_ref (a bead diameter below its top
    layer, the 90th percentile of the beads' tops) and its pore volume below
    that plane, from a state the run kept."""
    ev, _, _ = _state(state)
    x, L = _centres(), float(ev.periods[0])
    z = float(np.percentile(x[:, 2] + 1.0, 90)) - 2.0
    h = np.clip(z - (x[:, 2] - 1.0), 0.0, 2.0)
    return z, L*L*z - float(np.sum(np.pi*h*h*(3.0 - h)/3.0))


def pc64(runs):
    """The capillary pressure against the saturation for the periodic-64 set:
    from corpus.py's output folder (--runs DIR: each run's JSON and its
    first kept state) into drainage_pc64.npz, or from that file."""
    path = os.path.join(STATIC, "drainage_pc64.npz")
    if runs:
        data = {}
        for f in sorted(glob.glob(os.path.join(runs, "c64_*.json"))):
            r = json.load(open(f))
            name = r["name"].split("_h")[0][4:]                     # s0_t15
            _, P = pores(os.path.join(runs, "frames", r["name"], "step0000"))
            rows = [(min(V + T, P)/P, -p, fell) for _, V, p, T, _, fell, _ in r["curve"]]
            data[name] = np.array(rows + [(np.nan, float(r["reached"]), 0)])
        np.savez_compressed(path, **data)
    data = dict(np.load(path))
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]                     # seeds 0, 1, 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, theta in zip(axes, ("15", "20", "25")):
        for seed, colour in enumerate(colours):
            rows = data.get(f"s{seed}_t{theta}")
            if rows is None:
                continue
            S, Pc, reached = rows[:-1, 0], rows[:-1, 1], rows[-1, 1]
            ax.plot(S, Pc, "-", color=colour, lw=1.2, label=f"seed {seed}")
            if reached:
                ax.plot(S[-1], Pc[-1], "o", color=colour, ms=5)
        ax.set_title(f"contact angle {theta} degrees", fontsize=10)
        ax.set_xlim(1.02, 0.2)
        ax.set_xlabel("saturation")
        ax.grid(color="#e4e3de", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("capillary pressure (tension / bead radius)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "drainage_pc64.png"), dpi=150)
    plt.close(fig)
    print(f"wrote drainage_pc64.png ({len(data)} runs)")



# ---- the two beads in a periodic cell (periodic.md) ---------------------------

def periodic_tiles(p, corners, body, n=2):
    """The liquid's facets (corners at one image each, moved into the cell),
    the cell tiled n x n, and the beads with it."""
    import pyvista as pv
    L = periodic_helpers.L
    c = np.array(corners, float)
    c[:, :, :2] -= (np.floor(c[:, 0, :2]/L)*L)[:, None, :]
    x = c.reshape(-1, 3)
    f = np.arange(len(x)).reshape(-1, 3)
    for a in range(n):
        for b in range(n):
            liquid(p, x + [a*L, b*L, 0], f, body)
            for centre in periodic_helpers.BEADS:
                p.add_mesh(pv.Sphere(radius=1.0, center=np.add(centre, [a*L, b*L, 0]),
                                     theta_resolution=36, phi_resolution=36), **BEAD)
    p.add_mesh(pv.Plane(center=(n*L/2, n*L/2, 0), direction=(0, 0, 1), i_size=n*L,
                        j_size=n*L), color="#dddddd")
    p.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, 4.2)), style="wireframe", color="#555555")


PERIODIC_CAMERA = [(15.0, -9.0, 9.5), (3.0, 3.0, 1.6), (0, 0, 1)]


def periodic():
    warnings.simplefilter("ignore")
    ev = periodic_helpers.two_beads(theta=20.0)
    steps, meshes = [], []
    for s in recipes.arclength(ev, ev.body(1), -0.5, 200, topology=True, outlet="bottom"):
        m = ev.mesh()
        steps.append(s)
        meshes.append((np.array(m.facet_corners()), np.asarray(m.facet_bodies).max(axis=1)))
        if ev.on_constraint("bottom").any():
            break
    V = np.array([s.value for s in steps])
    T = np.array([s.trapped for s in steps])
    cut = int(np.flatnonzero(T > 0)[0])
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1500, 600), border=False)
    for col, i in enumerate((cut - 3, cut, len(steps) - 1)):
        p.subplot(0, col)
        periodic_tiles(p, *meshes[i])
        p.add_text(f"volume {V[i]:.2f}" + (f", ring {T[i]:.3f}" if T[i] else ""), font_size=11)
        p.camera_position = PERIODIC_CAMERA
    p.screenshot(os.path.join(STATIC, "periodic_ring_shapes.png"))
    p.close()

    def draw(pl, i):
        periodic_tiles(pl, *meshes[i])
        pl.add_text(f"volume {V[i]:.2f}" + (f", ring {T[i]:.3f}" if T[i] else ""),
                    font_size=11)
    movie(os.path.join(STATIC, "periodic_ring.mp4"), range(len(steps)), draw,
          PERIODIC_CAMERA)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="*", default=["drop", "pore", "ring", "pack", "packs"])
    ap.add_argument("--series", help="series.py's output folder (pack)")
    ap.add_argument("--logs", help="corpus.py's output folder of the wide set (packs)")
    ap.add_argument("--snaps", help="bench/topology/snapshots.py's snapshots of a run (pack64)")
    ap.add_argument("--runs", help="corpus.py's output folder of the periodic-64 set (pc64)")
    args = ap.parse_args()
    for part in args.parts:
        if part == "pack":
            if not args.series:
                sys.exit("pack needs --series DIR")
            pack(args.series)
        elif part == "pack64":
            if not args.snaps:
                sys.exit("pack64 needs --snaps DIR")
            pack64(args.snaps)
        elif part == "packs":
            packs(args.logs)
        elif part == "pc64":
            pc64(args.runs)
        else:
            globals()[part]()

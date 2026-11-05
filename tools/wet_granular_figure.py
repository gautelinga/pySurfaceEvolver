"""Draw the pictures and the movie of docs/wet_granular.md. Needs matplotlib,
PyVista (off-screen rendering) and imageio-ffmpeg (the movie).

    OMP_NUM_THREADS=1 python tools/wet_granular_figure.py [part ...] [--runs OUT ...]

Parts (both by default):

  pore   the pore between four beads (contact angle 20, 0.03 apart) wetted from
         a bridge at each of its six contacts: wet_pore.npz (the run's steps),
         wet_pore.png (the pressure and each bead's pull into the pore against
         the volume) and wet_pore_shapes.png (ten minutes)
  pack   beads in a periodic cell wetted from bridges at their contacts, from
         bench/topology/wet.py's output folders (--runs OUT ..., into
         wet_pack.npz, or from that file): wet_pack.png (the pressure, the
         liquid's pieces and the beads' forces against the saturation); with
         --film NAME (a run in the first folder, kept with --frames) also
         wet_pack.mp4 and wet_pack_shapes.png

Everything goes into docs/_static.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysurfaceevolver import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
sys.path.insert(0, HERE)
from packing_helpers import bridges  # noqa: E402
import drainage_topology_figure as D  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
SEEDS = ("#2a78d6", "#eb6834", "#1baf7a")     # (the runs' colours, as drainage_pc64.png's)
INK, MUTED = "#333333", "#888888"


def _axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


# ---- the pore between four beads --------------------------------------------

def pore_centres(contact=0.03):
    d = (2 + contact)/np.sqrt(2)
    return d/2*np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def pore_run(theta=20.0, ds=0.004, vmax=1.6):
    """The run's steps (volume, pressure, mean pull into the pore, fell, the
    changes made) and the mesh at each."""
    X = pore_centres()
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    ev = bridges(X, pairs, theta)
    g = X.mean(axis=0)
    rows, meshes = [], []
    for s in recipes.arclength(ev, ev.body(1), ds, 3000, topology=True, trap=False,
                               remesh=0.06):
        pull = np.mean([ev.wall_force(f"bead{k + 1}") @ (g - X[k])/np.linalg.norm(g - X[k])
                        for k in range(4)])
        m = ev.mesh()
        rows.append((s.value, s.pressures[0], pull, bool(s.fell), len(s.changes or ())))
        meshes.append((np.array(m.vertices), np.array(m.facets)))
        if s.value > vmax:
            break
    return np.array(rows, float), meshes


def pore():
    import pyvista as pv
    rows, meshes = pore_run()
    np.savez(os.path.join(STATIC, "wet_pore.npz"), rows=rows)
    V, p, pull, fell = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3] > 0
    # (the steps the run went on from: a back-up's steps replaced by the later ones)
    keep = np.ones(len(V), bool)
    for i in range(len(V) - 1, 0, -1):
        if keep[i]:
            keep[:i] &= ~(V[:i] > V[i] + 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for ax, y, label in ((axs[0], p, "pressure"), (axs[1], pull, "pull into the pore, per bead")):
        ax.plot(V[keep], y[keep], color=SEEDS[0], linewidth=2)
        ax.plot(V[keep & fell], y[keep & fell], "o", color=SEEDS[1], markersize=5,
                markeredgecolor="white", markeredgewidth=1, label="after a fall")
        ax.set_xlabel("liquid volume (bead radius$^3$)", color=INK)
        ax.set_ylabel(label, color=INK)
        _axes(ax)
    axs[0].legend(frameon=False)
    fig.savefig(os.path.join(STATIC, "wet_pore.png"), dpi=150)
    plt.close(fig)
    # the shapes: six bridges, just before the pore fills, filled, at the end
    k = np.flatnonzero(keep)
    jump = max((i for i in k[1:] if fell[i]), key=lambda i: p[i - 1] - p[i])
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, shape=(1, 4), window_size=(2400, 650), border=False)
    for col, i in enumerate((k[0], jump - 1, jump, k[-1])):
        pl.subplot(0, col)
        D.liquid(pl, *meshes[i])
        for c in pore_centres():
            pl.add_mesh(pv.Sphere(radius=1.0, center=c, theta_resolution=48,
                                  phi_resolution=48), **D.BEAD)
        pl.add_text(f"volume {V[i]:.2f}, pressure {p[i]:.1f}", font_size=12)
        pl.camera_position = [(5.0, 4.0, 3.0), (0, 0, 0), (0, 0, 1)]
    pl.screenshot(os.path.join(STATIC, "wet_pore_shapes.png"))
    pl.close()
    print("wrote wet_pore.npz, wet_pore.png and wet_pore_shapes.png")


# ---- a periodic pack ----------------------------------------------------------

def pack_data(runs):
    """{name: (rows, median force)} from wet.py's folders, kept in wet_pack.npz."""
    path = os.path.join(STATIC, "wet_pack.npz")
    if not runs:
        z = np.load(path, allow_pickle=True)
        return z["data"].item()
    data = {}
    for out in runs:
        for meta in sorted(glob.glob(os.path.join(out, "w*_s*_t*.json"))):
            name = os.path.basename(meta)[:-5]
            z = np.load(os.path.join(out, f"{name}.npz"))
            rows, F = z["rows"], z["forces"]
            data[name] = dict(rows=rows, force=np.median(np.linalg.norm(F, axis=2), axis=1),
                              meta=json.load(open(meta)))
    np.savez(path, data=np.array(data, dtype=object))
    return data


def pack(runs, film=None):
    data = pack_data(runs)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for k, (name, d) in enumerate(sorted(data.items())):
        r = d["rows"]
        S, p, count = r[:, 2], r[:, 3], r[:, 4]
        # (the steps the run went on from, as in pore())
        keep = np.ones(len(S), bool)
        for i in range(len(S) - 1, 0, -1):
            if keep[i]:
                keep[:i] &= ~(S[:i] > S[i] + 1e-12)
        m = d["meta"]
        label = f"{m['beads']} beads, seed {m['seed']}"
        colour = SEEDS[k % len(SEEDS)]
        axs[0].plot(S[keep], p[keep], color=colour, linewidth=1.8, label=label)
        axs[1].plot(S[keep], count[keep], color=colour, linewidth=1.8)
        axs[2].plot(S[keep], d["force"][keep], color=colour, linewidth=1.8)
    for ax, label in zip(axs, ("pressure", "pieces of liquid", "force on a bead (median)")):
        ax.set_xlabel("saturation", color=INK)
        ax.set_ylabel(label, color=INK)
        _axes(ax)
    axs[1].set_yscale("log")
    axs[0].legend(frameon=False)
    fig.savefig(os.path.join(STATIC, "wet_pack.png"), dpi=150)
    plt.close(fig)
    print("wrote wet_pack.npz and wet_pack.png")
    if film:
        pack_film(runs[0], film)


def pack_film(out, name, frames=240):
    """The run at ``frames`` equally spaced saturations (each the step nearest)."""
    import pyvista as pv
    z = np.load(os.path.join(out, f"{name}.npz"))
    beads = np.load(os.path.join(out, f"{name}_beads.npz"))
    centres, L = beads["centres"], float(beads["L"])
    rows = z["rows"]
    steps, S = rows[:, 0].astype(int), rows[:, 2]
    held = {}                                   # saturation -> the last step there
    for st, s in zip(steps, S):
        held[round(float(s), 9)] = st
    sat = np.array(sorted(held))
    chosen = [held[sat[int(np.argmin(abs(sat - t)))]]
              for t in np.linspace(sat[0], sat[-1], frames)]
    at = {st: s for s, st in held.items()}
    top = float(centres[:, 2].max()) + 1.0
    cell = D.cell_beads(centres, L, top)
    focus = np.array([L/2, L/2, top/2])
    look = np.array([1.0, -0.75, 0.6])
    camera = [tuple(focus + 2.2*max(top, L)*look/np.linalg.norm(look)), tuple(focus), (0, 0, 1)]

    def scene(pl, st):
        c = np.load(os.path.join(out, "frames", name, f"s{st:05d}.npz"))["c"].astype(float)
        D.cell_scene(pl, c, None, cell, L, top)
        pl.add_text(f"saturation {at[st]:.2f}", font_size=12)

    D.movie(os.path.join(STATIC, "wet_pack.mp4"), chosen, scene, camera, fps=24, hold=48,
            window=(720, 720))
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 650), border=False)
    for col, st in enumerate((chosen[0], chosen[len(chosen)//2], chosen[-1])):
        pl.subplot(0, col)
        scene(pl, st)
        pl.camera_position = camera
    pl.screenshot(os.path.join(STATIC, "wet_pack_shapes.png"))
    pl.close()
    print("wrote wet_pack.mp4 and wet_pack_shapes.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="*", default=["pore", "pack"])
    ap.add_argument("--runs", nargs="*", default=[], help="wet.py's output folders (pack)")
    ap.add_argument("--film", help="a run in the first --runs folder, kept with --frames (pack)")
    a = ap.parse_args()
    for part in a.parts:
        if part == "pore":
            pore()
        elif part == "pack":
            pack(a.runs, a.film)
        else:
            raise SystemExit(f"no part {part!r}")

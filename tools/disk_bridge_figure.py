"""Draw the pictures of docs/disk_bridge.md: liquid bridges between two equal
disks of radius 1, their contact lines pinned at the edges. Needs matplotlib,
PyVista (off-screen rendering) and SciPy; takes about ten minutes.

    OMP_NUM_THREADS=4 python tools/disk_bridge_figure.py [shapes]

writes docs/_static/disk_bridge.png (the stable region of slenderness and
reduced volume: the exact limits without gravity and at three Bond numbers,
with the page's pySE runs and the largest volumes tabulated by Slobozhanin
and Perales) and docs/_static/disk_bridge_shapes.png (four bridges just past
their limits, coloured by the unstable mode); with ``shapes``, only the
latter (a few minutes).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysurfaceevolver as pyse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
import disk_bridge_helpers as D  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
COLOURS = {0.0: "black", 0.2: "#1f77b4", 0.5: "#ff7f0e", 1.0: "#d62728"}
# the page's runs: (bond, slenderness, direction)
RUNS = ([(0.0, s, d) for s in (0.5, 1.5, 2.5) for d in ("less", "more")]
        + [(b, s, d) for b, s in ((0.2, 1.96), (0.5, 1.29), (1.0, 0.93)) for d in ("less", "more")])
TABLE = {0.2: (1.96, 3.865), 0.5: (1.29, 2.663), 1.0: (0.93, 2.124)}   # Slobozhanin and Perales
LIQUID = dict(smooth_shading=True, ambient=0.3, diffuse=0.7)


def stable_start(slenderness, bond):
    """A reduced volume with a stable axisymmetric bridge, or None."""
    for volume in (1.0, 1.3, 1.7, 2.2):
        try:
            q, H = D.start(slenderness, bond, volume)
        except RuntimeError:
            continue
        if min(D.Profile(q, H, bond).lowest()) > 0:
            return volume
    return None


def region(bond, slendernesses):
    """(slenderness, least volume, largest volume) of the stable bridges."""
    rows = []
    for s in slendernesses:
        v = stable_start(s, bond)
        if v is None:
            break
        try:
            lim = D.limits(s, bond, v)
        except RuntimeError:
            break
        rows.append((s, lim["less"][0], lim["more"][0]))
        print(f"  bond {bond} slenderness {s:.3f}: {rows[-1][1]:.4f} to {rows[-1][2]:.4f}",
              flush=True)
    return np.array(rows)


def run(bond, slenderness, direction):
    """The page's run: the limit, and the surface just past it with its
    unstable mode (points, triangles, normal displacement)."""
    ev = D.disk_bridge(slenderness, bond=bond)
    ev.relax(levels=1)
    ev.command("quadratic")
    volume, how = D.to_limit(ev, slenderness, -0.04 if direction == "less" else 0.15)
    mode = ev.modes(1).vectors[0]
    points, triangles, d = ev.mesh().tessellate(values=mode)
    return volume, how, (points, triangles, d)


def polydata(points, triangles):
    import pyvista as pv
    return pv.PolyData(np.asarray(points, float),
                       np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel())


def render_shapes(shapes):
    import pyvista as pv
    p = pv.Plotter(shape=(1, len(shapes)), off_screen=True, window_size=(330*len(shapes), 430),
                   border=False, lighting="three lights")
    p.set_background("white")
    for k, (label, slenderness, (points, triangles, d)) in enumerate(shapes):
        p.subplot(0, k)
        mesh = polydata(points, triangles).compute_normals(auto_orient_normals=True)
        u = (d*mesh.point_data["Normals"]).sum(axis=1)
        mesh["normal motion"] = u/abs(u).max()
        p.add_mesh(mesh, scalars="normal motion", cmap="coolwarm", clim=(-1, 1),
                   show_scalar_bar=False, **LIQUID)
        H = 2*slenderness
        for z in (0.0, H):
            p.add_mesh(pv.Disc(center=(0, 0, z), inner=0.0, outer=1.0, c_res=60),
                       color="#9a9a9a", ambient=0.4)
        p.add_text(label, position="upper_left", font_size=9, color="black")
        v = np.array([0.0, -1.0, 0.3])
        v /= np.linalg.norm(v)
        p.camera.focal_point = (0, 0, H/2)
        p.camera.position = tuple(np.array([0, 0, H/2]) + 12*v)
        p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True
        # half the view's height: the bridge's height and width, in a panel 330 x 430
        p.camera.parallel_scale = 0.55*max(np.ptp(points[:, 2]) + 0.6,
                                           (np.ptp(points[:, 0]) + 0.3)*430/330)
    p.screenshot(os.path.join(STATIC, "disk_bridge_shapes.png"))
    p.close()


def main():
    warnings.simplefilter("ignore", pyse.EvolverWarning)
    results, shapes = {}, {}
    for bond, s, direction in RUNS:
        volume, how, shape = run(bond, s, direction)
        results[bond, s, direction] = volume
        shapes[bond, s, direction] = shape
        print(f"pySE bond {bond} slenderness {s} {direction}: {volume:.4f} ({how})", flush=True)
    render_shapes([("slenderness 1.5,\nleast volume", 1.5, shapes[0.0, 1.5, "less"]),
                   ("slenderness 1.5,\nlargest volume", 1.5, shapes[0.0, 1.5, "more"]),
                   ("slenderness 2.5,\nleast volume", 2.5, shapes[0.0, 2.5, "less"]),
                   ("Bond 0.5, slenderness 1.29,\nleast volume", 1.29, shapes[0.5, 1.29, "less"])])
    if "shapes" in sys.argv[1:]:
        return

    curves = {0.0: region(0.0, np.linspace(0.45, 3.13, 18))}
    for bond in (0.2, 0.5, 1.0):
        curves[bond] = region(bond, np.linspace(0.45, 3.25, 29))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for bond, rows in curves.items():
        colour = COLOURS[bond]
        label = "no gravity" if bond == 0 else f"Bond {bond}"
        ax.plot(rows[:, 0], rows[:, 1], color=colour, label=label)
        ax.plot(rows[:, 0], rows[:, 2], color=colour)
        ax.plot([rows[-1, 0]]*2, rows[-1, 1:], color=colour, ls=":", lw=0.8)
    for (bond, s, direction), volume in results.items():
        ax.plot(s, volume, "o", ms=6, mfc="none", mew=1.4, color=COLOURS[bond])
    for bond, (s, volume) in TABLE.items():
        ax.plot(s, volume, "x", ms=8, mew=1.4, color=COLOURS[bond])
    ax.plot([], [], "ko", mfc="none", label="pySE, this page")
    ax.plot([], [], "kx", label="largest volume, Slobozhanin and Perales")
    ax.set_yscale("log")
    ax.set_xlabel("slenderness H/(2R)")
    ax.set_ylabel("reduced volume V/(πR²H)")
    ax.set_xlim(0.4, 3.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Stable liquid bridges between equal disks", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "disk_bridge.png"), dpi=110)



if __name__ == "__main__":
    main()

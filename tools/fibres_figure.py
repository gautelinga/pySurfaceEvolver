"""Draw the pictures of docs/fibres.md: where a barrel on a fibre rolls up and
how a clam shell's energy compares, the columns between two parallel fibres
against Princen's, and a column at two crossed fibres against the model of
Sauret et al.; then the surfaces. Needs matplotlib, PyVista (off-screen
rendering) and SciPy; takes about five minutes.

    OMP_NUM_THREADS=4 python tools/fibres_figure.py

writes docs/_static/fibres.png (the curves) and docs/_static/fibres_shapes.png.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysurfaceevolver import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
import fibres_helpers as H  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.2,
              smooth_shading=False, ambient=0.25, diffuse=0.75)
SOLID = dict(color="#c8c8c8", ambient=0.3, diffuse=0.7)


def snapshot(ev):
    m = ev.mesh()
    return m.vertices.copy(), m.facets.copy()


def barrel_run(theta):
    """(volume, sideways eigenvalue) as the barrel shrinks through its onset."""
    exact = H.barrel_onset(theta)["volume"]
    volumes = exact*np.linspace(1.3, 0.8, 11)
    ev = H.barrel_drop(theta, volumes[0])
    rows = [(step.value, H.shift(ev)) for step in recipes.continuation(ev, ev.body(1), volumes)]
    return np.array(rows), exact


def clamshell_run(theta, volumes):
    """(volume, clam shell energy - barrel energy), and the two surfaces at the
    middle volume."""
    ev = H.clamshell(theta, volumes[0], levels=2)
    rows, shapes = [], {}
    for step in recipes.continuation(ev, ev.body(1), volumes):
        rows.append((step.value, ev.energy - H.barrel(theta, step.value)["energy"]))
        if step.value == 30:
            shapes["clam shell"] = snapshot(ev)
    shapes["barrel"] = snapshot(H.barrel_drop(theta, 30.0))
    return np.array(rows), shapes


def column_run(d):
    """(area, pressure) of the long column's middle, by the balance."""
    ev = H.column_slice(d, 0.0)
    b = H.column_area(ev)
    gb = H.column_excess(ev, b)
    a, ga = b, gb
    while ga*gb > 0:
        a, ga = b, gb
        b *= 0.9 if gb > 0 else 1.1
        gb = H.column_excess(ev, b)
    for _ in range(5):
        c = b - gb*(b - a)/(gb - ga)
        gc = H.column_excess(ev, c)
        a, ga, b, gb = (a, ga, c, gc) if gc*ga < 0 else (c, gc, b, gb)
    return c, ev.bodies().pressure[0]


def crossed_run(delta, theta, volumes, keep):
    """(volume, half length, wrap angle) as the column grows, and its surface at
    the volumes in ``keep``."""
    ev = H.crossed(delta, 20.0, volumes[0])       # relaxed at 20 degrees, then lowered
    ev.relax(levels=1)
    for _ in recipes.continuation(ev, "theta", [a for a in (15, 10, 5, 2) if a > theta] + [theta]):
        pass
    rows, shapes = [], {}
    for volume in volumes:
        ev.body(1).target = volume
        ev.remesh(target=0.22, keep_boundary=False)
        ev.relax()
        x = ev.vertices
        rows.append((volume, (x[:, 0].max() - x[:, 0].min())/2, H.wrap_angle(ev, delta)))
        if volume in keep:
            shapes[volume] = snapshot(ev)
    return np.array(rows), shapes


def polydata(vertices, facets):
    import pyvista as pv
    return pv.PolyData(np.asarray(vertices, float),
                       np.hstack([np.full((len(facets), 1), 3), facets]).ravel())


def view(p, direction, focus, scale, up=(0, 0, 1)):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    p.camera.focal_point = tuple(focus)
    p.camera.position = tuple(np.asarray(focus) + 30*d)
    p.camera.up = up
    p.camera.parallel_projection = True
    p.camera.parallel_scale = scale


def main():
    warnings.simplefilter("ignore")
    barrels = {theta: barrel_run(theta) for theta in (30, 45, 60)}
    clam, clam_shapes = clamshell_run(30, [10, 15, 20, 25, 30, 35, 40, 50])
    gaps = (0.5, 0.75, 1.0, 1.3)
    columns = [column_run(d) for d in gaps]
    delta, theta = 30.0, 0.0
    volumes = (5, 10, 20, 30, 40, 50, 60, 70)
    crossing, crossing_shapes = crossed_run(delta, theta, volumes, keep=(30, 70))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    ax = axes[0]
    for (theta_b, (rows, exact)), colour in zip(barrels.items(), COLOURS):
        ax.plot(rows[:, 0]/exact, rows[:, 1], "o-", ms=3, color=colour, label=f"{theta_b} degrees")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(1, color="black", lw=0.5, ls=":")
    ax.set_xlabel("volume / exact onset volume")
    ax.set_ylabel("eigenvalue of the sideways shift")
    ax.legend(fontsize=8, title="contact angle", title_fontsize=8)
    ax.set_title("A barrel shrinking through its onset", fontsize=10)

    ax = axes[1]
    ax.plot(clam[:, 0], clam[:, 1], "o-", ms=3, color="black")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(H.barrel_onset(30)["volume"], color=COLOURS[0], ls=":",
               label="the barrel's onset")
    ax.set_xlabel("volume")
    ax.set_ylabel("clam shell energy - barrel energy")
    ax.legend(fontsize=8)
    ax.set_title("Clam shell against barrel (30 degrees)", fontsize=10)

    ax = axes[2]
    ds = np.linspace(0.3, np.sqrt(2) - 1e-6, 200)
    exact = [H.princen(d, 0.0) for d in ds]
    ax.plot(2*ds, [e["pressure"] for e in exact], color="black", label="Princen")
    ax.plot([2*d for d in gaps], [p for _, p in columns], "o", color=COLOURS[1], label="pySE")
    ax.axvline(2*np.sqrt(2), color="black", ls=":", lw=1)
    ax.set_xlabel("gap between the fibres")
    ax.set_ylabel("pressure in the column")
    twin = ax.twinx()
    twin.plot(2*ds, [e["area"] for e in exact], color="#7f7f7f", lw=1)
    twin.plot([2*d for d in gaps], [a for a, _ in columns], "o", mfc="none", color="#7f7f7f")
    twin.set_ylabel("area of its cross-section (grey)", color="#7f7f7f")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("A long column between parallel fibres (0 degrees)", fontsize=10)

    ax = axes[3]
    model = H.crossed_column(delta, theta)
    vs = np.linspace(1, model["volume_max"], 60)
    ax.plot(vs, [H.crossed_column(delta, theta, v)["half_length"] for v in vs], color="black",
            label="model (Sauret et al.)")
    ax.plot(crossing[:, 0], crossing[:, 1], "o", ms=3, color=COLOURS[1], label="pySE")
    ax.set_xlabel("volume")
    ax.set_ylabel("half length of the column")
    twin = ax.twinx()
    twin.plot(crossing[:, 0], crossing[:, 2], "--", color="#d62728", lw=1)
    twin.axhline(180, color="#d62728", lw=0.5)
    twin.set_ylabel("how far round the fibres it wets (degrees)", color="#d62728")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Crossed fibres ({delta:.0f} degrees, contact angle {theta:.0f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "fibres.png"), dpi=110)
    plt.close(fig)

    import pyvista as pv
    p = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1320, 440), border=False,
                   lighting="three lights")
    p.set_background("white")
    p.subplot(0, 0)
    for (name, mesh), dx in zip(clam_shapes.items(), (-4.5, 3.5)):
        v = mesh[0] + [dx, 0, 0]
        p.add_mesh(polydata(v, mesh[1]), **LIQUID)
        p.add_mesh(pv.Cylinder(center=(dx, 0, 0), direction=(0, 0, 1), radius=1.0, height=9.0,
                               resolution=60), **SOLID)
    p.add_text("a clam shell and a barrel holding the same volume\n(30 on a fibre of radius 1, "
               "30 degrees)", position="upper_left", font_size=9, color="black")
    view(p, (0.0, -1.0, 0.3), (0, 0, 0), 5.5)
    h = np.radians(delta)/2
    for col, volume in ((1, 30), (2, 70)):
        p.subplot(0, col)
        p.add_mesh(polydata(*crossing_shapes[volume]), **LIQUID)
        for z, s in ((1.0, 1), (-1.0, -1)):
            p.add_mesh(pv.Cylinder(center=(0, 0, z), direction=(np.cos(h), s*np.sin(h), 0),
                                   radius=1.0, height=26.0, resolution=60), opacity=0.45, **SOLID)
        p.add_text(f"liquid at crossed fibres, volume {volume}", position="upper_left",
                   font_size=9, color="black")
        view(p, (0.2, -0.5, 1.0), (0, 0, 0), {30: 9.0, 70: 12.0}[volume])
    p.screenshot(os.path.join(STATIC, "fibres_shapes.png"))
    p.close()
    print("wrote", os.path.normpath(STATIC) + "/fibres*.png")


if __name__ == "__main__":
    main()

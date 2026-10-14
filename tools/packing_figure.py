"""Draw the pictures of docs/packing.md: the runs of its three examples (a
throat in a layer of spheres, a trimer between three beads, a pore full of
liquid) and their surfaces where each stops. Needs matplotlib, PyVista
(off-screen rendering) and SciPy; takes about three minutes.

    OMP_NUM_THREADS=4 python tools/packing_figure.py

writes docs/_static/packing.png (the curves) and docs/_static/packing_shapes.png.
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
import packing_helpers as H  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c")
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.2,
              smooth_shading=False, ambient=0.25, diffuse=0.75)


def snapshot(ev):
    m = ev.mesh()
    return m.vertices.copy(), m.facets.copy()


def throat_run(theta, gap=0.0):
    """(volume, curvature) along the drain, and the surface at the entry."""
    ev = H.throat_layer(theta, gap=gap)
    rows, best = [], (0.0, None)
    for step in recipes.arclength(ev, ev.body(1), -0.004, 200):
        c = -step.pressures[0]
        rows.append((step.value, c))
        if c > best[0]:
            best = (c, snapshot(ev))
        if ev.vertices[:, 2].min() < -0.6:
            break
    return np.array(rows), best[1]


def trimer_run(gap, theta=20.0):
    """(volume, thickness) along the drain, the surface at the end, the centres."""
    ev, centres = H.trimer(gap, theta, 0.45)
    rows = []
    for step in recipes.arclength(ev, ev.body(1), -0.005, 300):
        h = H.throat_thickness(ev, centres)
        rows.append((step.value, h))
        if h < 0.02 or step.result.negative_modes:
            break
    return np.array(rows), snapshot(ev), centres


def pore_run(gap=0.02, theta=20.0):
    """(volume, curvature, four lowest eigenvalues) along the drain, the
    surface at the peak, the centres."""
    ev, centres = H.filled_pore(gap, theta)
    rows, best = [], (0.0, None)
    for step in recipes.arclength(ev, ev.body(1), -0.01, 100):
        c = -step.pressures[0]
        rows.append((step.value, c, *ev.modes(4).values))
        if c > best[0]:
            best = (c, snapshot(ev))
        if step.result.negative_modes:
            break
    return np.array(rows), best[1], centres


def polydata(vertices, facets):
    import pyvista as pv
    return pv.PolyData(np.asarray(vertices, float),
                       np.hstack([np.full((len(facets), 1), 3), facets]).ravel())


def balls(p, centres, opacity=1.0):
    import pyvista as pv
    for c in centres:
        p.add_mesh(pv.Sphere(radius=1.0, center=tuple(c), theta_resolution=90,
                             phi_resolution=90), color="#c8c8c8", ambient=0.3, diffuse=0.7,
                   opacity=opacity)


def view(p, direction, focus, scale, up=(0, 0, 1)):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    p.camera.focal_point = tuple(focus)
    p.camera.position = tuple(np.asarray(focus) + 8*d)
    p.camera.up = up
    p.camera.parallel_projection = True
    p.camera.parallel_scale = scale


def main():
    warnings.simplefilter("ignore")
    throats = {theta: throat_run(theta) for theta in (20.0, 40.0, 60.0)}
    trimers = {gap: trimer_run(gap) for gap in (0.09, 0.17, 0.25)}
    pore, pore_mesh, pore_centres = pore_run()
    layer, _ = throat_run(20.0, gap=0.02)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.9))
    for (theta, (rows, _)), colour in zip(throats.items(), COLOURS):
        ax1.plot(rows[:, 0], rows[:, 1], color=colour, label=f"{theta:.0f} degrees")
        ax1.axhline(H.msp_throat(theta), color=colour, ls=":", lw=1)
    ax1.set_xlabel("liquid volume above the throat plane")
    ax1.set_ylabel("curvature")
    ax1.legend(fontsize=8, title="contact angle (dotted: MSP)", title_fontsize=8)
    ax1.set_title("Gas passing a throat", fontsize=10)

    for (gap, (rows, _, _)), colour in zip(trimers.items(), COLOURS):
        ax2.plot(rows[:, 0], rows[:, 1], color=colour, label=f"gap {gap}")
        ax2.plot(rows[-1, 0], rows[-1, 1], "x", color=colour)
    ax2.set_xlabel("liquid volume")
    ax2.set_ylabel("thickness of the liquid in the throat")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.set_title("A trimer drained to its burst (20 degrees)", fontsize=10)

    ax3.plot(pore[:, 0], pore[:, 1], color="black", label="curvature")
    ax3.axhline(layer[:, 1].max(), color="black", ls=":", lw=1,
                label="the throat's entry curvature")
    ax3.set_xlabel("liquid volume")
    ax3.set_ylabel("curvature")
    ax3.invert_xaxis()
    twin = ax3.twinx()
    for k in range(4):
        twin.plot(pore[:, 0], pore[:, 2 + k], color="#d62728", lw=1,
                  ls="-" if k < 3 else "--")
    twin.axhline(0, color="#d62728", lw=0.5)
    twin.set_ylabel("lowest eigenvalues", color="#d62728")
    ax3.legend(fontsize=8, loc="lower left")
    ax3.set_title("A full pore drained (20 degrees, gap 0.02)", fontsize=10)
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "packing.png"), dpi=110)
    plt.close(fig)

    import pyvista as pv
    p = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1260, 440), border=False,
                   lighting="three lights")
    p.set_background("white")
    p.subplot(0, 0)
    L = 2.0
    layer_centres = np.array([[0, 0, 0], [L, 0, 0], [L/2, L*np.sqrt(3)/2, 0]])
    balls(p, layer_centres, opacity=0.3)
    p.add_mesh(polydata(*throats[40.0][1]), **LIQUID)
    p.add_text("the meniscus at the throat's entry curvature\n(40 degrees; from below, one prism)",
               position="upper_left", font_size=9, color="black")
    view(p, (0.25, -0.45, -1.0), layer_centres.mean(axis=0), 1.7)
    p.subplot(0, 1)
    rows, mesh, centres = trimers[0.17]
    balls(p, centres)
    p.add_mesh(polydata(*mesh), **LIQUID)
    p.add_text(f"the trimer as it bursts\n(gap 0.17, volume {rows[-1, 0]:.3f})",
               position="upper_left", font_size=9, color="black")
    view(p, (0.0, -0.5, 1.0), centres.mean(axis=0), 1.7)
    p.subplot(0, 2)
    balls(p, pore_centres, opacity=0.12)
    p.add_mesh(polydata(*pore_mesh), **LIQUID)
    p.add_text("the pore at the peak of the curvature", position="upper_left", font_size=9,
               color="black")
    view(p, (1.0, -0.6, 0.45), pore_centres.mean(axis=0), 1.6)
    p.screenshot(os.path.join(STATIC, "packing_shapes.png"))
    p.close()
    print("wrote", os.path.normpath(STATIC) + "/packing*.png")


if __name__ == "__main__":
    main()

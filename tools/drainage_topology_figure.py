"""Draw the pictures of docs/drainage_topology.md: one pore of a bead pack (four
unit beads, contact angle 20) drained with recipes.arclength(topology=True).
Needs matplotlib and PyVista (off-screen rendering); takes about a minute.

    OMP_NUM_THREADS=1 python tools/drainage_topology_figure.py

writes docs/_static/drainage_pore.png (the liquid's curvature against its
volume, the changes of topology marked) and drainage_pore_shapes.png (the
liquid just before and just after the jump).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
from packing_helpers import _pore_surface, _on  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
CENTRES = np.array([[-0.6123, 1.1001, 0.3757], [1.2879, 0.1557, 0.1317],
                    [-0.4315, -1.0784, 0.651], [-0.2441, -0.1774, -1.1584]])
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.2,
              smooth_shading=False, ambient=0.25, diffuse=0.75)


def run():
    """The page's run: the steps, and the mesh at each point."""
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
    steps, meshes = [], []
    for s in recipes.arclength(ev, ev.body(1), -0.01, 150, topology=True):
        m = ev.mesh()
        steps.append(s)
        meshes.append((np.array(m.vertices), np.array(m.facets)))
        if s.value < 0.2:
            break
    return steps, meshes


def trace(steps):
    V = np.array([s.value for s in steps])
    k = np.array([-s.pressures[0] for s in steps])
    fell = np.array([s.fell for s in steps])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(V, k, "-", color="#1f77b4", lw=1.2, label="arclength points")
    ax.plot(V[fell], k[fell], "o", color="#d62728", ms=5, label="after a change of topology")
    ax.invert_xaxis()
    ax.set_xlabel("liquid volume (drained to the right)")
    ax.set_ylabel("curvature of the liquid surface")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "drainage_pore.png"), dpi=150)


def shapes(steps, meshes):
    import pyvista as pv
    pv.OFF_SCREEN = True
    k = np.array([-s.pressures[0] for s in steps])
    jump = max((i for i, s in enumerate(steps) if s.fell), key=lambda i: k[i - 1] - k[i])
    p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1400, 700), border=False)
    for col, i in enumerate((jump - 1, jump)):
        p.subplot(0, col)
        x, f = meshes[i]
        p.add_mesh(pv.PolyData(x, np.c_[np.full(len(f), 3), f]), **LIQUID)
        for c in CENTRES:
            p.add_mesh(pv.Sphere(radius=1.0, center=c, theta_resolution=48, phi_resolution=48),
                       color="sandybrown", opacity=0.25)
        p.add_text(f"volume {steps[i].value:.2f}, curvature {k[i]:.1f}", font_size=12)
        p.camera_position = [(5.0, 4.0, 3.0), (0, 0, 0), (0, 0, 1)]
    p.screenshot(os.path.join(STATIC, "drainage_pore_shapes.png"))


if __name__ == "__main__":
    steps, meshes = run()
    trace(steps)
    shapes(steps, meshes)
    print("wrote docs/_static/drainage_pore.png and drainage_pore_shapes.png")
